"""The generated reporter PKI: a ten-year P-256 CA and a one-year reporter-edge leaf under it.

Requirement 5.1 and design.md "PKI and durable state": default mode generates a P-256 reporter CA under ``/data`` with a
ten-year lifetime and a one-year reporter-edge leaf renewed 30 days before expiry. Routine leaf renewal keeps the leaf's
private key and SPKI, so reporters never need to re-enroll. Reporters trust the exact SPKI hash of the CA
certificate (the CA pin); a leaf pin is unsupported (requirement 5.2).

The CA's signing key and the leaf's key live in **separate** owner-only directories and documents. The privilege-
separated reporter edge (task 5.1) is meant to hold only its leaf key, never the CA signing key, so no record here
contains both. The leaf record names the CA it chains to (its pin and generation), and every load of the leaf verifies
that chain against the CA record, so a leaf that does not belong to the stored CA fails closed.

This module owns the durable records and certificate issuance only. It does not choose the certificate mode, schedule
renewal (task 12.1), perform a CA transition (requirement 5.4), read operator files, or run any listener.

Creation is two steps: the CA, then its first leaf. A crash between them leaves a CA and no leaf, which
:meth:`ReporterPkiStore.load_leaf` reports as ``IDENTITY_MISSING``; :meth:`ReporterPkiStore.create_leaf` finishes it.

Leaf host names are DNS names in canonical lowercase LDH form (no IP addresses). The reporter's server-identity host is
supplied by the caller; this module only records and certifies it.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable, Mapping, Sequence
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
    require_aware,
    scalar_bytes,
    spki_der,
    spki_identifier,
)
from remote_energy_meter.protocol_v1 import JsonValue, ProtocolViolation, decode_base64url, encode_base64url
from remote_energy_meter.protocol_v1.encoding import decode_opaque_base64url
from remote_energy_meter.secure_store import SecureStoreError, StoredDocument, StoreFault, ensure_private_directory

__all__ = [
    "CA_DOCUMENT_NAME",
    "CA_LIFETIME",
    "CERTIFICATE_BACKDATE",
    "LEAF_DOCUMENT_NAME",
    "LEAF_LIFETIME",
    "MAX_DNS_NAMES",
    "RENEW_BEFORE_EXPIRY",
    "ReporterCa",
    "ReporterLeaf",
    "ReporterPkiStore",
]

CA_DOCUMENT_NAME: Final = "reporter-ca.json"
LEAF_DOCUMENT_NAME: Final = "reporter-leaf.json"
CA_LIFETIME: Final = timedelta(days=3650)
"""Validity of the reporter CA: ten years (requirement 5.1)."""
LEAF_LIFETIME: Final = timedelta(days=365)
"""Validity of each reporter-edge leaf: one year (requirement 5.1), never beyond the CA's own expiry."""
RENEW_BEFORE_EXPIRY: Final = timedelta(days=30)
"""A leaf needs renewal once less than this remains (requirement 5.1)."""
CERTIFICATE_BACKDATE: Final = timedelta(minutes=5)
"""How far ``not_before`` is set before issuance, so a peer whose clock runs slightly behind still accepts it."""
MAX_DNS_NAMES: Final = 8

_CA_STORE_ID: Final = "reporter-ca"
_LEAF_STORE_ID: Final = "reporter-leaf"
_SCHEMA_VERSION: Final = 1
_MAX_BYTES: Final = 16384
_MAX_CERTIFICATE_BYTES: Final = 4096
_CA_KEYS: Final = frozenset({"ca_generation", "certificate_der", "private_scalar", "spki_sha256"})
_LEAF_KEYS: Final = frozenset(
    {"ca_generation", "ca_spki_sha256", "certificate_der", "dns_names", "private_scalar", "spki_sha256"}
)
_CA_COMMON_NAME: Final = "Remote Energy Meter reporter CA"
_LEAF_COMMON_NAME: Final = "Remote Energy Meter reporter edge"
_LABEL: Final = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


def _canonical_dns_names(names: Sequence[str]) -> tuple[str, ...]:
    """Validate and sort DNS names: lowercase LDH labels, no IP-like or trailing-dot names, 1 to 8 distinct."""
    if isinstance(names, str) or not 1 <= len(names) <= MAX_DNS_NAMES:
        raise ValueError(f"between 1 and {MAX_DNS_NAMES} DNS names are required")
    for name in names:
        labels = name.split(".")
        if len(name) > 253 or not all(_LABEL.fullmatch(label) for label in labels) or labels[-1].isdigit():
            raise ValueError("a DNS name must be lowercase LDH labels and must not look like an IP address")
    if len(set(names)) != len(names):
        raise ValueError("DNS names must be distinct")
    return tuple(sorted(names))


def _certificate_b64(certificate: x509.Certificate) -> str:
    der = certificate.public_bytes(serialization.Encoding.DER)
    return encode_base64url(der, exact_length=len(der))


@dataclass(frozen=True, slots=True)
class ReporterCa:
    """The loaded reporter CA. Its signing key sits in an opaque holder that never appears in ``repr``."""

    spki_sha256: str
    """The CA pin reporters verify: canonical base64url SHA-256 of the DER SubjectPublicKeyInfo."""
    ca_generation: int
    certificate_der: bytes = field(repr=False)
    not_before: datetime
    not_after: datetime
    private_scalar: InitVar[bytes]
    _scalar: PrivateScalar = field(init=False, repr=False, compare=False)

    def __post_init__(self, private_scalar: bytes) -> None:
        object.__setattr__(self, "_scalar", PrivateScalar(private_scalar))

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("a reporter CA cannot be pickled or copied")

    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        raise TypeError("a reporter CA cannot be pickled or copied")

    def private_key(self) -> ec.EllipticCurvePrivateKey:
        """Return the CA signing key. It must never leave the core."""
        return self._scalar.key()


@dataclass(frozen=True, slots=True)
class ReporterLeaf:
    """The loaded reporter-edge leaf. Its private key sits in an opaque holder that never appears in ``repr``."""

    spki_sha256: str
    ca_spki_sha256: str
    ca_generation: int
    dns_names: tuple[str, ...]
    certificate_der: bytes = field(repr=False)
    not_before: datetime
    not_after: datetime
    private_scalar: InitVar[bytes]
    _scalar: PrivateScalar = field(init=False, repr=False, compare=False)

    def __post_init__(self, private_scalar: bytes) -> None:
        object.__setattr__(self, "_scalar", PrivateScalar(private_scalar))

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("a reporter leaf cannot be pickled or copied")

    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        raise TypeError("a reporter leaf cannot be pickled or copied")

    def private_key(self) -> ec.EllipticCurvePrivateKey:
        """Return the leaf's TLS private key. Hand it only to the reporter TLS edge."""
        return self._scalar.key()

    def private_key_pkcs8_der(self) -> bytes:
        """Return the leaf key as unencrypted PKCS#8 DER, for the reporter edge's own owner-only key file."""
        return self.private_key().private_bytes(
            serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )

    def needs_renewal(self, now: datetime) -> bool:
        """Whether less than :data:`RENEW_BEFORE_EXPIRY` of the leaf's validity remains at ``now``."""
        return require_aware(now, "now") >= self.not_after - RENEW_BEFORE_EXPIRY


class ReporterPkiStore:
    """Durable, authenticated storage of the generated reporter CA and leaf.

    Args:
        ca_directory: Directory of the CA record. It holds the CA signing key, so only the core may read it.
        leaf_directory: A different directory for the leaf record, the only key material the reporter edge needs.
        keys: The state-authentication key store whose key tags both records.
        clock: Returns the current time as a timezone-aware ``datetime``; injectable for tests.
    """

    def __init__(
        self,
        ca_directory: Path,
        leaf_directory: Path,
        *,
        keys: StateAuthenticationKeyStore,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if ca_directory == leaf_directory:
            raise ValueError("the CA and the leaf must be stored in different directories")
        self._ca_directory = ca_directory
        self._leaf_directory = leaf_directory
        self._clock = clock
        self._ca_document = AuthenticatedDocumentStore(
            ca_directory,
            CA_DOCUMENT_NAME,
            store_id=_CA_STORE_ID,
            schema_version=_SCHEMA_VERSION,
            max_bytes=_MAX_BYTES,
            keys=keys,
        )
        self._leaf_document = AuthenticatedDocumentStore(
            leaf_directory,
            LEAF_DOCUMENT_NAME,
            store_id=_LEAF_STORE_ID,
            schema_version=_SCHEMA_VERSION,
            max_bytes=_MAX_BYTES,
            keys=keys,
        )

    @property
    def ca_path(self) -> Path:
        """The CA document's path."""
        return self._ca_document.path

    @property
    def leaf_path(self) -> Path:
        """The leaf document's path."""
        return self._leaf_document.path

    # -- creation and renewal ---------------------------------------------------------------------

    def create(self, dns_names: Sequence[str]) -> ReporterLeaf:
        """Create the CA and its first leaf, once.

        Raises:
            ValueError: ``dns_names`` is invalid.
            SecureStoreError: ``IDENTITY_EXISTS`` when a CA already exists (it is never replaced); any fault of the
                underlying stores. Use :meth:`create_leaf` to finish a creation interrupted after the CA.
        """
        names = _canonical_dns_names(dns_names)
        self.create_ca()
        return self.create_leaf(names)

    def create_ca(self) -> ReporterCa:
        """Generate and store the CA, exclusively.

        Raises:
            SecureStoreError: ``IDENTITY_EXISTS`` when a CA already exists, including one created while this call
                runs; ``KEY_MISSING`` without a state-authentication key; any other store fault.
        """
        ensure_private_directory(self._ca_directory)
        if self._ca_document.load() is not None:
            raise SecureStoreError(StoreFault.IDENTITY_EXISTS, f"{self.ca_path} already exists and is never replaced")
        private_key = ec.generate_private_key(ec.SECP256R1())
        certificate = self._issue_ca(private_key)
        record: dict[str, JsonValue] = {
            "ca_generation": 1,
            "certificate_der": _certificate_b64(certificate),
            "private_scalar": encode_base64url(scalar_bytes(private_key), exact_length=SCALAR_BYTES),
            "spki_sha256": spki_identifier(private_key.public_key()),
        }
        try:
            self._ca_document.create(record)
        except SecureStoreError as error:
            if error.fault is StoreFault.CONCURRENT_MODIFICATION:
                detail = f"{self.ca_path} was created by another writer and is never replaced"
                raise SecureStoreError(StoreFault.IDENTITY_EXISTS, detail) from None
            raise
        return self._ca_from(record)

    def create_leaf(self, dns_names: Sequence[str]) -> ReporterLeaf:
        """Generate and store the first leaf under the stored CA, exclusively.

        Raises:
            ValueError: ``dns_names`` is invalid.
            SecureStoreError: ``IDENTITY_MISSING`` without a CA; ``IDENTITY_EXISTS`` when a leaf already exists (it is
                never replaced); ``CERTIFICATE_EXPIRED`` when the CA has expired; any other store fault.
        """
        names = _canonical_dns_names(dns_names)
        ca = self.load_ca()
        ensure_private_directory(self._leaf_directory)
        if self._leaf_document.load() is not None:
            raise SecureStoreError(StoreFault.IDENTITY_EXISTS, f"{self.leaf_path} already exists and is never replaced")
        private_key = ec.generate_private_key(ec.SECP256R1())
        record = self._leaf_record(ca, private_key, names)
        try:
            self._leaf_document.create(record)
        except SecureStoreError as error:
            if error.fault is StoreFault.CONCURRENT_MODIFICATION:
                detail = f"{self.leaf_path} was created by another writer and is never replaced"
                raise SecureStoreError(StoreFault.IDENTITY_EXISTS, detail) from None
            raise
        return self._leaf_from(record, ca)

    def renew_leaf(self, dns_names: Sequence[str] | None = None) -> ReporterLeaf:
        """Issue a new leaf certificate under the same leaf key; the leaf SPKI never changes.

        Args:
            dns_names: New host names for the certificate, or ``None`` to keep the current ones. The key and SPKI are
                kept either way.

        Raises:
            ValueError: ``dns_names`` is invalid.
            SecureStoreError: ``IDENTITY_MISSING`` when no leaf (or CA) exists; ``CERTIFICATE_EXPIRED`` when the CA
                has expired; ``IDENTITY_INCONSISTENT`` when the stored records disagree; any other store fault.
        """
        names = None if dns_names is None else _canonical_dns_names(dns_names)
        ca = self.load_ca()
        # The update loads the state key first; the identity is the more specific fault when both are missing.
        if self._leaf_document.load() is None:
            raise SecureStoreError(StoreFault.IDENTITY_MISSING, f"{self.leaf_path} does not exist")

        def reissue(current: StoredDocument | None) -> Mapping[str, JsonValue]:
            if current is None:
                raise SecureStoreError(StoreFault.IDENTITY_MISSING, f"{self.leaf_path} does not exist")
            leaf = self._leaf_from(current.record, ca)
            return self._leaf_record(ca, leaf.private_key(), leaf.dns_names if names is None else names)

        return self._leaf_from(self._leaf_document.update(reissue).record, ca)

    # -- loading ----------------------------------------------------------------------------------

    def load_ca(self) -> ReporterCa:
        """Read and fully verify the CA.

        Raises:
            SecureStoreError: ``IDENTITY_MISSING`` when none exists; ``IDENTITY_INCONSISTENT`` for an authentic but
                inconsistent record; ``MALFORMED`` for a record of the wrong shape; any fault of the underlying store.
        """
        document = self._ca_document.load()
        if document is None:
            raise SecureStoreError(StoreFault.IDENTITY_MISSING, f"{self.ca_path} does not exist")
        return self._ca_from(document.record)

    def load_leaf(self) -> ReporterLeaf:
        """Read the leaf and verify it, and that it chains to the stored CA.

        Raises:
            SecureStoreError: ``IDENTITY_MISSING`` when no CA or leaf exists; ``IDENTITY_INCONSISTENT`` when the leaf
                does not belong to the stored CA or its parts disagree; any fault of :meth:`load_ca` or the store.
        """
        ca = self.load_ca()
        document = self._leaf_document.load()
        if document is None:
            raise SecureStoreError(StoreFault.IDENTITY_MISSING, f"{self.leaf_path} does not exist")
        return self._leaf_from(document.record, ca)

    # -- issuance ---------------------------------------------------------------------------------

    def _now(self) -> datetime:
        return require_aware(self._clock(), "clock").astimezone(UTC)

    def _issue_ca(self, private_key: ec.EllipticCurvePrivateKey) -> x509.Certificate:
        now = self._now()
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _CA_COMMON_NAME)])
        public_key = private_key.public_key()
        return (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(public_key)
            .serial_number(secrets.randbits(63) + 1)
            .not_valid_before(now - CERTIFICATE_BACKDATE)
            .not_valid_after(now + CA_LIFETIME)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
            .sign(private_key, hashes.SHA256())
        )

    def _leaf_record(
        self, ca: ReporterCa, leaf_key: ec.EllipticCurvePrivateKey, names: tuple[str, ...]
    ) -> Mapping[str, JsonValue]:
        now = self._now()
        if now >= ca.not_after:
            raise SecureStoreError(StoreFault.CERTIFICATE_EXPIRED, "the reporter CA has expired; no leaf can be issued")
        ca_key = ca.private_key()
        leaf_public = leaf_key.public_key()
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _LEAF_COMMON_NAME)])
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(x509.load_der_x509_certificate(ca.certificate_der).subject)
            .public_key(leaf_public)
            .serial_number(secrets.randbits(63) + 1)
            .not_valid_before(now - CERTIFICATE_BACKDATE)
            .not_valid_after(min(now + LEAF_LIFETIME, ca.not_after))
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
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(n) for n in names]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(leaf_public), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        return {
            "ca_generation": ca.ca_generation,
            "ca_spki_sha256": ca.spki_sha256,
            "certificate_der": _certificate_b64(certificate),
            "dns_names": list(names),
            "private_scalar": encode_base64url(scalar_bytes(leaf_key), exact_length=SCALAR_BYTES),
            "spki_sha256": spki_identifier(leaf_public),
        }

    # -- decoding and verification ----------------------------------------------------------------

    def _certificate(self, where: Path, encoded: str) -> x509.Certificate:
        try:
            return x509.load_der_x509_certificate(
                decode_opaque_base64url(encoded, minimum=1, maximum=_MAX_CERTIFICATE_BYTES)
            )
        except (ProtocolViolation, ValueError):
            raise SecureStoreError(StoreFault.MALFORMED, f"{where} holds an undecodable certificate") from None

    def _key(self, where: Path, encoded: str) -> ec.EllipticCurvePrivateKey:
        try:
            scalar = decode_base64url(encoded, exact_length=SCALAR_BYTES)
            return ec.derive_private_key(int.from_bytes(scalar, "big"), ec.SECP256R1())
        except (ProtocolViolation, ValueError):
            raise SecureStoreError(StoreFault.MALFORMED, f"{where} holds an undecodable private key") from None

    def _ca_from(self, record: Mapping[str, JsonValue]) -> ReporterCa:
        where = self.ca_path
        if set(record) != _CA_KEYS:
            raise SecureStoreError(StoreFault.MALFORMED, f"{where} is not a reporter CA record")
        scalar_text, cert_text = record["private_scalar"], record["certificate_der"]
        pin, generation = record["spki_sha256"], record["ca_generation"]
        if (
            type(scalar_text) is not str
            or type(cert_text) is not str
            or type(pin) is not str
            or type(generation) is not int
            or generation < 1
        ):
            raise SecureStoreError(StoreFault.MALFORMED, f"{where} has fields of the wrong type")
        try:
            decode_base64url(pin, exact_length=HASH_BYTES)
        except ProtocolViolation:
            raise SecureStoreError(StoreFault.MALFORMED, f"{where} has a malformed pin") from None
        certificate = self._certificate(where, cert_text)
        private_key = self._key(where, scalar_text)
        public_key = private_key.public_key()
        certificate_key = certificate.public_key()
        if (
            spki_identifier(public_key) != pin
            or not isinstance(certificate_key, ec.EllipticCurvePublicKey)
            or spki_der(certificate_key) != spki_der(public_key)
            or certificate.subject != certificate.issuer
            or not isinstance(certificate.signature_hash_algorithm, hashes.SHA256)
            or not self._is_ca(certificate)
        ):
            raise SecureStoreError(StoreFault.IDENTITY_INCONSISTENT, f"{where}: CA certificate, key, and pin disagree")
        self._verify_signature(where, certificate, public_key)
        return ReporterCa(
            spki_sha256=pin,
            ca_generation=generation,
            certificate_der=certificate.public_bytes(serialization.Encoding.DER),
            not_before=certificate.not_valid_before_utc,
            not_after=certificate.not_valid_after_utc,
            private_scalar=scalar_bytes(private_key),
        )

    def _leaf_from(self, record: Mapping[str, JsonValue], ca: ReporterCa) -> ReporterLeaf:
        where = self.leaf_path
        if set(record) != _LEAF_KEYS:
            raise SecureStoreError(StoreFault.MALFORMED, f"{where} is not a reporter leaf record")
        scalar_text, cert_text = record["private_scalar"], record["certificate_der"]
        pin, ca_pin = record["spki_sha256"], record["ca_spki_sha256"]
        generation, raw_names = record["ca_generation"], record["dns_names"]
        if (
            type(scalar_text) is not str
            or type(cert_text) is not str
            or type(pin) is not str
            or type(ca_pin) is not str
            or type(generation) is not int
            or generation < 1
            or type(raw_names) is not list
            or not all(type(name) is str for name in raw_names)
        ):
            raise SecureStoreError(StoreFault.MALFORMED, f"{where} has fields of the wrong type")
        try:
            decode_base64url(pin, exact_length=HASH_BYTES)
            decode_base64url(ca_pin, exact_length=HASH_BYTES)
            names = _canonical_dns_names([str(name) for name in raw_names])
        except (ProtocolViolation, ValueError):
            raise SecureStoreError(StoreFault.MALFORMED, f"{where} has a malformed pin or host name") from None
        if names != tuple(raw_names):
            raise SecureStoreError(StoreFault.MALFORMED, f"{where} lists host names that are not canonical")
        certificate = self._certificate(where, cert_text)
        private_key = self._key(where, scalar_text)
        public_key = private_key.public_key()
        certificate_key = certificate.public_key()
        if (
            spki_identifier(public_key) != pin
            or not isinstance(certificate_key, ec.EllipticCurvePublicKey)
            or spki_der(certificate_key) != spki_der(public_key)
            or self._is_ca(certificate)
            or not isinstance(certificate.signature_hash_algorithm, hashes.SHA256)
            or self._dns_names(certificate) != names
        ):
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{where}: leaf certificate, key, and pin disagree"
            )
        if ca_pin != ca.spki_sha256 or generation != ca.ca_generation:
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{where}: the leaf does not belong to the stored CA"
            )
        try:
            certificate.verify_directly_issued_by(x509.load_der_x509_certificate(ca.certificate_der))
        except (InvalidSignature, ValueError, TypeError):
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{where}: the leaf is not signed by the stored CA"
            ) from None
        return ReporterLeaf(
            spki_sha256=pin,
            ca_spki_sha256=ca_pin,
            ca_generation=generation,
            dns_names=names,
            certificate_der=certificate.public_bytes(serialization.Encoding.DER),
            not_before=certificate.not_valid_before_utc,
            not_after=certificate.not_valid_after_utc,
            private_scalar=scalar_bytes(private_key),
        )

    @staticmethod
    def _is_ca(certificate: x509.Certificate) -> bool:
        try:
            return certificate.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
        except x509.ExtensionNotFound:
            return False

    @staticmethod
    def _dns_names(certificate: x509.Certificate) -> tuple[str, ...]:
        try:
            san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        except x509.ExtensionNotFound:
            return ()
        return tuple(sorted(san.get_values_for_type(x509.DNSName)))

    @staticmethod
    def _verify_signature(where: Path, certificate: x509.Certificate, public_key: ec.EllipticCurvePublicKey) -> None:
        algorithm = certificate.signature_hash_algorithm
        if algorithm is None:
            raise SecureStoreError(StoreFault.IDENTITY_INCONSISTENT, f"{where}: the certificate has no hash algorithm")
        try:
            public_key.verify(certificate.signature, certificate.tbs_certificate_bytes, ec.ECDSA(algorithm))
        except (InvalidSignature, TypeError, ValueError):
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{where}: the certificate is not signed by its own key"
            ) from None
