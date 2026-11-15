"""The control listener's TLS identity: one P-256 key and a mode-independent self-signed certificate.

Requirement 7.5d: the dedicated control TLS key and certificate live under ``/data`` apart from the reporter CA, the
reporter-edge leaf, and operator files. The identity peers pin is the base64url SHA-256 of the exact DER
SubjectPublicKeyInfo (SPKI). Reporter certificate mode, CA transitions, and public-PKI renewal never create, sign, or
change it, and routine certificate renewal reuses the key, so it never changes the SPKI.

This module owns only that identity's durable record:

* :meth:`ControlIdentityStore.create` generates the key and the first certificate, once.
* :meth:`ControlIdentityStore.renew` issues a fresh certificate under the *same* key and SPKI.
* :meth:`ControlIdentityStore.load` verifies, on every call, that the certificate is self-signed by the stored key and
  that the stored identifier is the hash of that key's SPKI, and fails closed otherwise.

Planned SPKI rotation, recovery revocation, and the generations they advance belong to the control-authority work. The
``spki_generation`` stored here is the generation of this key (always ``1`` until a rotation replaces it).

The certificate carries no subject alternative name: control peers verify the exact SPKI pin, not a host name, and the
endpoint's address is deliberately not recorded. It is a TLS server certificate (``serverAuth``) valid for one year,
backdated five minutes against clock skew; :meth:`ControlIdentity.needs_renewal` turns true 30 days before expiry,
matching the reporter-edge leaf policy.

The document is an :class:`~remote_energy_meter.authenticated_store.AuthenticatedDocumentStore` record, so an edited
or replayed record fails authentication, and it sits in its own owner-only directory.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from remote_energy_meter.authenticated_store import AuthenticatedDocumentStore, StateAuthenticationKeyStore
from remote_energy_meter.protocol_v1 import JsonValue, ProtocolViolation, decode_base64url, encode_base64url
from remote_energy_meter.secure_store import SecureStoreError, StoreFault, ensure_private_directory

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
_SCALAR_BYTES: Final = 32
_HASH_BYTES: Final = 32
_RECORD_KEYS: Final = frozenset({"certificate_der", "private_scalar", "spki_generation", "spki_sha256"})
_COMMON_NAME: Final = "Remote Energy Meter control"


def _spki_der(public_key: ec.EllipticCurvePublicKey) -> bytes:
    return public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)


def _spki_identifier(public_key: ec.EllipticCurvePublicKey) -> str:
    return encode_base64url(hashlib.sha256(_spki_der(public_key)).digest(), exact_length=_HASH_BYTES)


def _encode_blob(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_blob(text: str) -> bytes:
    """Decode unpadded base64url strictly: only the canonical encoding of the bytes is accepted."""
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        raise ValueError("not base64url") from None
    if _encode_blob(raw) != text:
        raise ValueError("not canonical base64url")
    return raw


@dataclass(frozen=True, slots=True)
class ControlIdentity:
    """The loaded control identity. The private key never appears in ``repr``."""

    spki_sha256: str
    """Canonical base64url SHA-256 of the DER SubjectPublicKeyInfo: the pin peers verify."""
    spki_generation: int
    certificate_der: bytes = field(repr=False)
    not_before: datetime
    not_after: datetime
    _private_scalar: bytes = field(repr=False)

    def private_key(self) -> ec.EllipticCurvePrivateKey:
        """Return the control TLS private key. Hand it only to the control TLS edge."""
        return ec.derive_private_key(int.from_bytes(self._private_scalar, "big"), ec.SECP256R1())

    def private_key_pkcs8_der(self) -> bytes:
        """Return the private key as unencrypted PKCS#8 DER, for the control edge's own owner-only key file."""
        return self.private_key().private_bytes(
            serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )

    def needs_renewal(self, now: datetime) -> bool:
        """Whether less than :data:`RENEW_BEFORE_EXPIRY` of the certificate's validity remains at ``now``."""
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

        Raises:
            SecureStoreError: ``IDENTITY_EXISTS`` when an identity already exists (it is never replaced); any fault
                of the underlying store, including ``KEY_MISSING`` without a state-authentication key.
        """
        ensure_private_directory(self._directory)
        if self._document.load() is not None:
            raise SecureStoreError(StoreFault.IDENTITY_EXISTS, f"{self.path} already exists and is never replaced")
        private_key = ec.generate_private_key(ec.SECP256R1())
        certificate = self._issue(private_key)
        self._document.save(self._record(private_key, certificate, spki_generation=1))
        return self._identity(private_key, certificate, spki_generation=1)

    def renew(self) -> ControlIdentity:
        """Issue a new certificate under the same key and store it; the SPKI and its generation do not change.

        Raises:
            SecureStoreError: ``IDENTITY_MISSING`` when none exists; any fault of :meth:`load`.
        """
        current = self.load()
        private_key = current.private_key()
        certificate = self._issue(private_key)
        renewed = self._identity(private_key, certificate, spki_generation=current.spki_generation)
        if renewed.spki_sha256 != current.spki_sha256:
            raise SecureStoreError(StoreFault.IDENTITY_INCONSISTENT, "renewal would change the control SPKI")
        self._document.save(self._record(private_key, certificate, spki_generation=current.spki_generation))
        return renewed

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
        now = self._clock().astimezone(UTC)
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
        scalar = private_key.private_numbers().private_value.to_bytes(_SCALAR_BYTES, "big")
        return {
            "certificate_der": _encode_blob(certificate.public_bytes(serialization.Encoding.DER)),
            "private_scalar": encode_base64url(scalar, exact_length=_SCALAR_BYTES),
            "spki_generation": spki_generation,
            "spki_sha256": _spki_identifier(private_key.public_key()),
        }

    def _identity(
        self, private_key: ec.EllipticCurvePrivateKey, certificate: x509.Certificate, *, spki_generation: int
    ) -> ControlIdentity:
        scalar = private_key.private_numbers().private_value.to_bytes(_SCALAR_BYTES, "big")
        return ControlIdentity(
            spki_sha256=_spki_identifier(private_key.public_key()),
            spki_generation=spki_generation,
            certificate_der=certificate.public_bytes(serialization.Encoding.DER),
            not_before=certificate.not_valid_before_utc,
            not_after=certificate.not_valid_after_utc,
            _private_scalar=scalar,
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
            scalar = decode_base64url(encoded_scalar, exact_length=_SCALAR_BYTES)
            decode_base64url(encoded_spki, exact_length=_HASH_BYTES)
            certificate = x509.load_der_x509_certificate(_decode_blob(encoded_cert))
            private_key = ec.derive_private_key(int.from_bytes(scalar, "big"), ec.SECP256R1())
        except (ProtocolViolation, ValueError):
            raise SecureStoreError(
                StoreFault.MALFORMED, f"{self.path} holds an undecodable key or certificate"
            ) from None
        public_key = private_key.public_key()
        certificate_key = certificate.public_key()
        if (
            _spki_identifier(public_key) != encoded_spki
            or not isinstance(certificate_key, ec.EllipticCurvePublicKey)
            or _spki_der(certificate_key) != _spki_der(public_key)
            or certificate.subject != certificate.issuer
        ):
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{self.path}: certificate, key, and identifier disagree"
            )
        algorithm = certificate.signature_hash_algorithm
        if algorithm is None:
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{self.path}: the certificate has no hash algorithm"
            )
        try:
            public_key.verify(certificate.signature, certificate.tbs_certificate_bytes, ec.ECDSA(algorithm))
        except (InvalidSignature, TypeError, ValueError):
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{self.path}: the certificate is not signed by the stored key"
            ) from None
        return self._identity(private_key, certificate, spki_generation=spki_generation)
