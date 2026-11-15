"""The control TLS identity: one key, a self-signed certificate, and an SPKI pin that renewal never changes."""

from __future__ import annotations

import base64
import hashlib
import stat
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID

from remote_energy_meter import control_identity
from remote_energy_meter.authenticated_store import AuthenticatedDocumentStore, StateAuthenticationKeyStore
from remote_energy_meter.control_identity import (
    CERTIFICATE_BACKDATE,
    CERTIFICATE_LIFETIME,
    DOCUMENT_NAME,
    RENEW_BEFORE_EXPIRY,
    ControlIdentity,
    ControlIdentityStore,
)
from remote_energy_meter.protocol_v1 import encode_base64url, render_canonical_json
from remote_energy_meter.secure_store import SecureStoreError, StoreFault

T0 = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def keys(tmp_path: Path) -> StateAuthenticationKeyStore:
    store = StateAuthenticationKeyStore(tmp_path / "keys")
    store.create()
    return store


@pytest.fixture
def clock() -> Clock:
    return Clock(T0)


@pytest.fixture
def store(tmp_path: Path, keys: StateAuthenticationKeyStore, clock: Clock) -> ControlIdentityStore:
    return ControlIdentityStore(tmp_path / "control", keys=keys, clock=clock)


def fault_of(operation: Callable[[], object]) -> StoreFault:
    with pytest.raises(SecureStoreError) as caught:
        operation()
    return caught.value.fault


def independent_pin(certificate_der: bytes) -> str:
    """Recompute the pin from the certificate alone: base64url(SHA-256(DER SubjectPublicKeyInfo)), unpadded."""
    certificate = x509.load_der_x509_certificate(certificate_der)
    spki = certificate.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return base64.urlsafe_b64encode(hashlib.sha256(spki).digest()).rstrip(b"=").decode("ascii")


def replace_record(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore, **changes: Any
) -> ControlIdentityStore:
    """Re-save the identity record, correctly tagged, with fields altered: authentic but inconsistent.

    Returns a fresh store over the same directory, as a restarted add-on would read it; the original instance would
    rightly report the second writer.
    """
    document = AuthenticatedDocumentStore(
        store.path.parent, DOCUMENT_NAME, store_id="control-identity", schema_version=1, max_bytes=8192, keys=keys
    )
    loaded = document.load()
    assert loaded is not None
    record = dict(loaded.record)
    record.update(changes)
    document.save(record)
    return ControlIdentityStore(store.path.parent, keys=keys)


def another_certificate_der(tmp_path: Path, keys: StateAuthenticationKeyStore, clock: Clock) -> bytes:
    """The certificate of an unrelated, genuine control identity."""
    return ControlIdentityStore(tmp_path / "unrelated", keys=keys, clock=clock).create().certificate_der


# --------------------------------------------------------------------------------------------------
# creation and loading
# --------------------------------------------------------------------------------------------------


def test_create_stores_an_owner_only_identity_that_reloads_identically(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore, tmp_path: Path, clock: Clock
) -> None:
    created = store.create()
    loaded = ControlIdentityStore(tmp_path / "control", keys=keys, clock=clock).load()

    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert loaded == created
    assert loaded.private_key().private_numbers() == created.private_key().private_numbers()
    assert loaded.spki_generation == 1


def test_the_pin_is_the_unpadded_base64url_sha256_of_the_der_spki(store: ControlIdentityStore) -> None:
    identity = store.create()

    assert identity.spki_sha256 == independent_pin(identity.certificate_der)
    assert len(identity.spki_sha256) == 43
    assert "=" not in identity.spki_sha256


def test_the_certificate_is_a_self_signed_server_certificate_for_the_stored_key(store: ControlIdentityStore) -> None:
    identity = store.create()
    certificate = x509.load_der_x509_certificate(identity.certificate_der)

    certificate.verify_directly_issued_by(certificate)
    assert certificate.subject == certificate.issuer
    certificate_key = certificate.public_key()
    assert isinstance(certificate_key, ec.EllipticCurvePublicKey)
    assert certificate_key.public_numbers() == identity.private_key().public_key().public_numbers()
    constraints = certificate.extensions.get_extension_for_class(x509.BasicConstraints)
    assert constraints.critical and constraints.value.ca is False
    usage = certificate.extensions.get_extension_for_class(x509.KeyUsage).value
    assert usage.digital_signature and not usage.key_cert_sign and not usage.key_encipherment
    assert list(certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value) == [
        ExtendedKeyUsageOID.SERVER_AUTH
    ]
    with pytest.raises(x509.ExtensionNotFound):
        certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName)


def test_validity_is_one_year_backdated_five_minutes(store: ControlIdentityStore) -> None:
    identity = store.create()

    assert identity.not_before == T0 - CERTIFICATE_BACKDATE == T0 - timedelta(minutes=5)
    assert identity.not_after == T0 + CERTIFICATE_LIFETIME == T0 + timedelta(days=365)


def test_a_missing_identity_is_reported_and_never_created_by_load(store: ControlIdentityStore) -> None:
    assert fault_of(store.load) is StoreFault.IDENTITY_MISSING
    assert fault_of(store.renew) is StoreFault.IDENTITY_MISSING
    assert not store.path.exists()


def test_an_existing_identity_is_never_replaced(store: ControlIdentityStore) -> None:
    store.create()
    before = store.path.read_bytes()

    assert fault_of(store.create) is StoreFault.IDENTITY_EXISTS
    assert store.path.read_bytes() == before


def test_creating_without_a_state_authentication_key_fails_closed(tmp_path: Path, clock: Clock) -> None:
    orphan = ControlIdentityStore(
        tmp_path / "control", keys=StateAuthenticationKeyStore(tmp_path / "nokeys"), clock=clock
    )

    assert fault_of(orphan.create) is StoreFault.KEY_MISSING
    assert not orphan.path.exists()


def test_two_identities_have_different_pins(tmp_path: Path, keys: StateAuthenticationKeyStore, clock: Clock) -> None:
    first = ControlIdentityStore(tmp_path / "a", keys=keys, clock=clock).create()
    second = ControlIdentityStore(tmp_path / "b", keys=keys, clock=clock).create()

    assert first.spki_sha256 != second.spki_sha256


def test_the_private_key_is_never_in_repr(store: ControlIdentityStore) -> None:
    identity = store.create()
    scalar = identity.private_key().private_numbers().private_value

    rendered = repr(identity)

    assert str(scalar) not in rendered
    assert scalar.to_bytes(32, "big").hex() not in rendered
    assert "certificate_der" not in rendered
    assert identity.spki_sha256 in rendered


def test_the_key_exports_as_pkcs8_der_for_the_control_edge(store: ControlIdentityStore) -> None:
    identity = store.create()

    loaded = serialization.load_der_private_key(identity.private_key_pkcs8_der(), password=None)

    assert isinstance(loaded, ec.EllipticCurvePrivateKey)
    assert loaded.private_numbers() == identity.private_key().private_numbers()


# --------------------------------------------------------------------------------------------------
# renewal
# --------------------------------------------------------------------------------------------------


def test_renewal_keeps_the_key_the_pin_and_the_generation_and_issues_a_new_certificate(
    store: ControlIdentityStore, clock: Clock
) -> None:
    created = store.create()
    clock.now = T0 + timedelta(days=340)

    renewed = store.renew()

    assert renewed.spki_sha256 == created.spki_sha256 == independent_pin(renewed.certificate_der)
    assert renewed.spki_generation == created.spki_generation == 1
    assert renewed.private_key().private_numbers() == created.private_key().private_numbers()
    assert renewed.certificate_der != created.certificate_der
    assert x509.load_der_x509_certificate(renewed.certificate_der).serial_number != (
        x509.load_der_x509_certificate(created.certificate_der).serial_number
    )
    assert renewed.not_after == clock.now + CERTIFICATE_LIFETIME
    assert store.load() == renewed


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (timedelta(0), False),
        (CERTIFICATE_LIFETIME - RENEW_BEFORE_EXPIRY - timedelta(seconds=1), False),
        (CERTIFICATE_LIFETIME - RENEW_BEFORE_EXPIRY, True),
        (CERTIFICATE_LIFETIME - RENEW_BEFORE_EXPIRY + timedelta(seconds=1), True),
        (CERTIFICATE_LIFETIME, True),
        (CERTIFICATE_LIFETIME * 2, True),
    ],
)
def test_renewal_is_needed_exactly_thirty_days_before_expiry(
    store: ControlIdentityStore, elapsed: timedelta, expected: bool
) -> None:
    identity = store.create()

    assert identity.needs_renewal(T0 + elapsed) is expected


def test_renewal_never_changes_the_pin_across_many_renewals(store: ControlIdentityStore, clock: Clock) -> None:
    pin = store.create().spki_sha256
    for month in range(1, 6):
        clock.now = T0 + timedelta(days=30 * month)
        assert store.renew().spki_sha256 == pin
    assert store.load().spki_sha256 == pin


# --------------------------------------------------------------------------------------------------
# tampering and inconsistency
# --------------------------------------------------------------------------------------------------


def test_an_edited_record_fails_authentication(store: ControlIdentityStore, keys: StateAuthenticationKeyStore) -> None:
    store.create()
    envelope = store.path.read_bytes()
    store.path.write_bytes(envelope.replace(b'"spki_generation":1', b'"spki_generation":2'))

    assert fault_of(ControlIdentityStore(store.path.parent, keys=keys).load) is StoreFault.AUTHENTICATION_FAILED


def test_an_identifier_that_is_not_the_hash_of_the_key_is_inconsistent(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore
) -> None:
    store.create()
    fresh = replace_record(store, keys, spki_sha256=encode_base64url(b"\x00" * 32, exact_length=32))

    assert fault_of(fresh.load) is StoreFault.IDENTITY_INCONSISTENT


def test_a_certificate_for_another_key_is_inconsistent(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore, clock: Clock, tmp_path: Path
) -> None:
    store.create()
    foreign = base64.urlsafe_b64encode(another_certificate_der(tmp_path, keys, clock)).rstrip(b"=").decode("ascii")
    fresh = replace_record(store, keys, certificate_der=foreign)

    assert fault_of(fresh.load) is StoreFault.IDENTITY_INCONSISTENT


def test_a_private_key_that_does_not_match_the_certificate_is_inconsistent(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore
) -> None:
    store.create()
    other = ec.generate_private_key(ec.SECP256R1()).private_numbers().private_value.to_bytes(32, "big")
    fresh = replace_record(store, keys, private_scalar=encode_base64url(other, exact_length=32))

    assert fault_of(fresh.load) is StoreFault.IDENTITY_INCONSISTENT


@pytest.mark.parametrize(
    "changes",
    [
        {"private_scalar": "short"},
        {"private_scalar": encode_base64url(b"\x00" * 32, exact_length=32)},  # not a valid scalar
        {"spki_sha256": "not-a-hash"},
        {"certificate_der": "!!!!"},
        {"certificate_der": "AAAA"},
        {"spki_generation": 0},
        {"spki_generation": "1"},
        {"spki_generation": True},
    ],
)
def test_undecodable_or_mistyped_fields_are_malformed(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore, changes: dict[str, Any]
) -> None:
    store.create()
    fresh = replace_record(store, keys, **changes)

    assert fault_of(fresh.load) is StoreFault.MALFORMED


def test_a_record_with_missing_or_extra_fields_is_malformed(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore
) -> None:
    store.create()
    document = AuthenticatedDocumentStore(
        store.path.parent, DOCUMENT_NAME, store_id="control-identity", schema_version=1, max_bytes=8192, keys=keys
    )
    loaded = document.load()
    assert loaded is not None
    document.save({**loaded.record, "extra": 1})
    assert fault_of(ControlIdentityStore(store.path.parent, keys=keys).load) is StoreFault.MALFORMED

    document.save({"spki_generation": 1})
    assert fault_of(ControlIdentityStore(store.path.parent, keys=keys).load) is StoreFault.MALFORMED


def test_a_certificate_whose_signature_is_wrong_is_inconsistent(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore
) -> None:
    identity = store.create()
    der = bytearray(identity.certificate_der)
    der[-1] ^= 0x01  # inside the signature value
    fresh = replace_record(
        store, keys, certificate_der=base64.urlsafe_b64encode(bytes(der)).rstrip(b"=").decode("ascii")
    )

    assert fault_of(fresh.load) is StoreFault.IDENTITY_INCONSISTENT


def test_a_non_ec_certificate_is_inconsistent(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A certificate whose public key is not an EC key is refused, whatever else it claims."""
    identity = store.create()
    certificate = x509.load_der_x509_certificate(identity.certificate_der)

    class NotEC:
        pass

    monkeypatch.setattr(type(certificate), "public_key", lambda self: NotEC(), raising=False)

    assert fault_of(store.load) is StoreFault.IDENTITY_INCONSISTENT


def test_the_identity_lives_only_in_its_own_directory(
    store: ControlIdentityStore, tmp_path: Path, keys: StateAuthenticationKeyStore
) -> None:
    store.create()
    store.renew()

    assert sorted(entry.name for entry in (tmp_path / "control").iterdir()) == [DOCUMENT_NAME]
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["control", "keys"]


def test_the_module_exports_its_public_names() -> None:
    for name in control_identity.__all__:
        assert hasattr(control_identity, name)
    assert ControlIdentity.__slots__ is not None  # frozen slotted dataclass


def test_the_renderer_and_record_keys_are_the_documented_set(store: ControlIdentityStore) -> None:
    store.create()
    envelope = render_canonical_json({}, max_depth=1)
    assert envelope == b"{}"
    assert b'"spki_sha256"' in store.path.read_bytes()
