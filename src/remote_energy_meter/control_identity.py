"""The control listener's TLS identity: one P-256 key and a mode-independent self-signed certificate.

Requirement 7.5d: the dedicated control TLS key and certificate live under ``/data`` apart from the reporter CA, the
reporter-edge leaf, and operator files. The identity peers pin is the base64url SHA-256 of the exact DER
SubjectPublicKeyInfo (SPKI). Reporter certificate mode, CA transitions, and public-PKI renewal never create, sign, or
change it, and routine certificate renewal reuses the key, so it never changes the SPKI.

This module owns only that identity's durable record:

* :meth:`ControlIdentityStore.create` generates the key and the first certificate, once, writing exclusively so a
  concurrent creator's identity is never overwritten (it gets ``IDENTITY_EXISTS``).
* :meth:`ControlIdentityStore.renew` issues a fresh certificate under the *same* key and SPKI, reading and replacing
  the record in one locked update of the authenticated store, so the key and generation come from the record being
  replaced.
* :meth:`ControlIdentityStore.load` verifies, on every call, that the certificate is self-signed by the stored key and
  that the stored identifier is the hash of that key's SPKI, and fails closed otherwise.

Planned SPKI rotation, recovery revocation, and the generations they advance belong to the control-authority work. The
``spki_generation`` stored here is the generation of this key (always ``1`` until planned rotation exists and
replaces it).

The private key is never exposed for export beyond :meth:`ControlIdentity.private_key` and
:meth:`ControlIdentity.private_key_pkcs8_der`: pickling, copying, and ``dataclasses.asdict`` of an identity are
refused with ``TypeError``, and the key never appears in ``repr`` or ``str``.

The certificate carries no subject alternative name: control peers verify the exact SPKI pin, not a host name, and the
endpoint's address is deliberately not recorded. It is a TLS server certificate (``serverAuth``) valid for one year,
backdated five minutes against clock skew; :meth:`ControlIdentity.needs_renewal` turns true 30 days before expiry,
matching the reporter-edge leaf policy.

The document is an :class:`~remote_energy_meter.authenticated_store.AuthenticatedDocumentStore` record, so an edited
or replayed record fails authentication, and it sits in its own owner-only directory.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from remote_energy_meter.authenticated_store import AuthenticatedDocumentStore, StateAuthenticationKeyStore
from remote_energy_meter.key_material import (
    HASH_BYTES,
    SCALAR_BYTES,
    PrivateScalar,
    scalar_bytes,
    spki_der,
    spki_identifier,
)
from remote_energy_meter.protocol_v1 import JsonValue, ProtocolViolation, decode_base64url, encode_base64url
from remote_energy_meter.protocol_v1.encoding import decode_opaque_base64url
from remote_energy_meter.secure_store import SecureStoreError, StoredDocument, StoreFault, ensure_private_directory

__all__ = [
    "CERTIFICATE_BACKDATE",
    "CERTIFICATE_LIFETIME",
    "DOCUMENT_NAME",
    "RENEW_BEFORE_EXPIRY",
    "ControlIdentity",
    "ControlIdentityStore",
]

DOCUMENT_NAME: Final = "control-identity.json"
CERTIFICATE_LIFETIME: Final = timedelta(days=365)
"""Validity of each control certificate."""
CERTIFICATE_BACKDATE: Final = timedelta(minutes=5)
"""How far ``not_before`` is set before issuance, so a peer whose clock runs slightly behind still accepts it."""
RENEW_BEFORE_EXPIRY: Final = timedelta(days=30)
"""A certificate needs renewal once less than this remains."""

_STORE_ID: Final = "control-identity"
_SCHEMA_VERSION: Final = 1
_MAX_BYTES: Final = 8192
_MAX_CERTIFICATE_BYTES: Final = 4096
_RECORD_KEYS: Final = frozenset({"certificate_der", "private_scalar", "spki_generation", "spki_sha256"})
_COMMON_NAME: Final = "Remote Energy Meter control"


@dataclass(frozen=True, slots=True)
class ControlIdentity:
    """The loaded control identity. The private key sits in an opaque holder that never appears in ``repr``, and that
    refuses copying, ``dataclasses.asdict``, and pickling."""

    spki_sha256: str
    """Canonical base64url SHA-256 of the DER SubjectPublicKeyInfo: the pin peers verify."""
    spki_generation: int
    certificate_der: bytes = field(repr=False)
    not_before: datetime
    not_after: datetime
    private_scalar: InitVar[bytes]
    _scalar: PrivateScalar = field(init=False, repr=False, compare=False)

    def __post_init__(self, private_scalar: bytes) -> None:
        object.__setattr__(self, "_scalar", PrivateScalar(private_scalar))

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("a control identity cannot be pickled or copied")

    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        raise TypeError("a control identity cannot be pickled or copied")

    def private_key(self) -> ec.EllipticCurvePrivateKey:
        """Return the control TLS private key. Hand it only to the control TLS edge."""
        return self._scalar.key()

    def private_key_pkcs8_der(self) -> bytes:
        """Return the private key as unencrypted PKCS#8 DER, for the control edge's own owner-only key file."""
        return self.private_key().private_bytes(
            serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )

    def needs_renewal(self, now: datetime) -> bool:
        """Whether less than :data:`RENEW_BEFORE_EXPIRY` of the certificate's validity remains at ``now``."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be a timezone-aware datetime")
        return now >= self.not_after - RENEW_BEFORE_EXPIRY


class ControlIdentityStore:
    """Durable, authenticated storage of the control TLS identity.

    Args:
        directory: A dedicated directory for this identity, separate from reporter PKI and operator files.
        keys: The state-authentication key store whose key tags the record.
        clock: Returns the current UTC time; injectable for tests.
    """

    def __init__(
        self,
        directory: Path,
        *,
        keys: StateAuthenticationKeyStore,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._directory = directory
        self._clock = clock
        self._document = AuthenticatedDocumentStore(
            directory,
            DOCUMENT_NAME,
            store_id=_STORE_ID,
            schema_version=_SCHEMA_VERSION,
            max_bytes=_MAX_BYTES,
            keys=keys,
        )

    @property
    def path(self) -> Path:
        """The identity document's path."""
        return self._document.path

    def create(self) -> ControlIdentity:
        """Generate the key and first certificate and store them, once.

        The record is written exclusively, so an identity another writer creates concurrently is never overwritten.

        Raises:
            SecureStoreError: ``IDENTITY_EXISTS`` when an identity already exists, including one created while this
                call runs (it is never replaced); any other fault of the underlying store, including ``KEY_MISSING``
                without a state-authentication key and the tamper fault of an existing record that does not verify.
        """
        ensure_private_directory(self._directory)
        if self._document.load() is not None:
            raise SecureStoreError(StoreFault.IDENTITY_EXISTS, f"{self.path} already exists and is never replaced")
        private_key = ec.generate_private_key(ec.SECP256R1())
        certificate = self._issue(private_key)
        try:
            self._document.create(self._record(private_key, certificate, spki_generation=1))
        except SecureStoreError as error:
            if error.fault is StoreFault.CONCURRENT_MODIFICATION:
                detail = f"{self.path} was created by another writer and is never replaced"
                raise SecureStoreError(StoreFault.IDENTITY_EXISTS, detail) from None
            raise
        return self._identity(private_key, certificate, spki_generation=1)

    def renew(self) -> ControlIdentity:
        """Issue a new certificate under the same key and store it; the SPKI and its generation do not change.

        The current record is read, verified, and replaced under one store lock, so the key and generation come from the
        very record being replaced and a record changed by another writer is never overwritten with stale values.

        Raises:
            SecureStoreError: ``IDENTITY_MISSING`` when none exists; ``IDENTITY_INCONSISTENT`` when the record's parts
                disagree or renewal would change the SPKI; any other fault of :meth:`load`.
        """

        # The update loads the key first, so without this check a missing identity would report KEY_MISSING when the
        # key is also missing. The identity is the more specific fault; the update re-checks under its lock.
        if self._document.load() is None:
            raise SecureStoreError(StoreFault.IDENTITY_MISSING, f"{self.path} does not exist")

        def reissue(current: StoredDocument | None) -> Mapping[str, JsonValue]:
            if current is None:
                raise SecureStoreError(StoreFault.IDENTITY_MISSING, f"{self.path} does not exist")
            identity = self._decode(current.record)
            private_key = identity.private_key()
            certificate = self._issue(private_key)
            renewed = self._identity(private_key, certificate, spki_generation=identity.spki_generation)
            if renewed.spki_sha256 != identity.spki_sha256:
                raise SecureStoreError(StoreFault.IDENTITY_INCONSISTENT, "renewal would change the control SPKI")
            return self._record(private_key, certificate, spki_generation=identity.spki_generation)

        return self._decode(self._document.update(reissue).record)

    def load(self) -> ControlIdentity:
        """Read and fully verify the identity.

        Raises:
            SecureStoreError: ``IDENTITY_MISSING`` when none exists; ``IDENTITY_INCONSISTENT`` when the record is
                authentic but its parts disagree; ``MALFORMED`` for a record of the wrong shape; any fault of the
                underlying authenticated store (``AUTHENTICATION_FAILED``, ``KEY_MISSING``, rollback, permissions).
        """
        document = self._document.load()
        if document is None:
            raise SecureStoreError(StoreFault.IDENTITY_MISSING, f"{self.path} does not exist")
        return self._decode(document.record)

    # -- internals --------------------------------------------------------------------------------

    def _issue(self, private_key: ec.EllipticCurvePrivateKey) -> x509.Certificate:
        clock_now = self._clock()
        if clock_now.tzinfo is None or clock_now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        now = clock_now.astimezone(UTC)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _COMMON_NAME)])
        return (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(private_key.public_key())
            .serial_number(secrets.randbits(63) + 1)
            .not_valid_before(now - CERTIFICATE_BACKDATE)
            .not_valid_after(now + CERTIFICATE_LIFETIME)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(private_key, hashes.SHA256())
        )

    def _record(
        self, private_key: ec.EllipticCurvePrivateKey, certificate: x509.Certificate, *, spki_generation: int
    ) -> Mapping[str, JsonValue]:
        certificate_der = certificate.public_bytes(serialization.Encoding.DER)
        return {
            "certificate_der": encode_base64url(certificate_der, exact_length=len(certificate_der)),
            "private_scalar": encode_base64url(scalar_bytes(private_key), exact_length=SCALAR_BYTES),
            "spki_generation": spki_generation,
            "spki_sha256": spki_identifier(private_key.public_key()),
        }

    def _identity(
        self, private_key: ec.EllipticCurvePrivateKey, certificate: x509.Certificate, *, spki_generation: int
    ) -> ControlIdentity:
        return ControlIdentity(
            spki_sha256=spki_identifier(private_key.public_key()),
            spki_generation=spki_generation,
            certificate_der=certificate.public_bytes(serialization.Encoding.DER),
            not_before=certificate.not_valid_before_utc,
            not_after=certificate.not_valid_after_utc,
            private_scalar=scalar_bytes(private_key),
        )

    def _decode(self, record: Mapping[str, JsonValue]) -> ControlIdentity:
        if set(record) != _RECORD_KEYS:
            raise SecureStoreError(StoreFault.MALFORMED, f"{self.path} is not a control identity record")
        encoded_scalar, encoded_cert = record["private_scalar"], record["certificate_der"]
        encoded_spki, spki_generation = record["spki_sha256"], record["spki_generation"]
        if (
            type(encoded_scalar) is not str
            or type(encoded_cert) is not str
            or type(encoded_spki) is not str
            or type(spki_generation) is not int
            or spki_generation < 1
        ):
            raise SecureStoreError(StoreFault.MALFORMED, f"{self.path} has fields of the wrong type")
        try:
            scalar = decode_base64url(encoded_scalar, exact_length=SCALAR_BYTES)
            decode_base64url(encoded_spki, exact_length=HASH_BYTES)
            certificate = x509.load_der_x509_certificate(
                decode_opaque_base64url(encoded_cert, minimum=1, maximum=_MAX_CERTIFICATE_BYTES)
            )
            private_key = ec.derive_private_key(int.from_bytes(scalar, "big"), ec.SECP256R1())
        except (ProtocolViolation, ValueError):
            raise SecureStoreError(
                StoreFault.MALFORMED, f"{self.path} holds an undecodable key or certificate"
            ) from None
        public_key = private_key.public_key()
        certificate_key = certificate.public_key()
        if (
            spki_identifier(public_key) != encoded_spki
            or not isinstance(certificate_key, ec.EllipticCurvePublicKey)
            or spki_der(certificate_key) != spki_der(public_key)
            or certificate.subject != certificate.issuer
        ):
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{self.path}: certificate, key, and identifier disagree"
            )
        algorithm = certificate.signature_hash_algorithm
        if not isinstance(algorithm, hashes.SHA256):
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{self.path}: the certificate is not signed with SHA-256"
            )
        try:
            constraints = certificate.extensions.get_extension_for_class(x509.BasicConstraints).value
        except x509.ExtensionNotFound:
            constraints = None
        if constraints is None or constraints.ca:
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{self.path}: the certificate is not an end-entity certificate"
            )
        try:
            public_key.verify(certificate.signature, certificate.tbs_certificate_bytes, ec.ECDSA(algorithm))
        except (InvalidSignature, TypeError, ValueError):
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{self.path}: the certificate is not signed by the stored key"
            ) from None
        return self._identity(private_key, certificate, spki_generation=spki_generation)
