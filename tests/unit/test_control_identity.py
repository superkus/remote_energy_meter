"""The control TLS identity: one key, a self-signed certificate, and an SPKI pin that renewal never changes."""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import json
import os
import pickle
import stat
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

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
from remote_energy_meter.protocol_v1 import encode_base64url
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


# --------------------------------------------------------------------------------------------------
# creation and renewal against concurrent writers
# --------------------------------------------------------------------------------------------------


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def test_create_never_overwrites_an_identity_another_writer_installs_after_the_check(
    tmp_path: Path, keys: StateAuthenticationKeyStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = ControlIdentityStore(tmp_path / "control", keys=keys, clock=clock)
    second = ControlIdentityStore(tmp_path / "control", keys=keys, clock=clock)
    real_open = os.open
    winner: list[ControlIdentity] = []
    winner_bytes: list[bytes] = []

    def race_before_file_creation(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if flags & os.O_CREAT and not winner:
            winner.append(None)  # type: ignore[arg-type]  # recursion guard: the winner's create opens a file too
            winner[0] = second.create()
            winner_bytes.append(second.path.read_bytes())
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", race_before_file_creation)
    fault = fault_of(first.create)
    monkeypatch.undo()

    assert winner_bytes, "the competing identity was never created"
    assert fault is StoreFault.IDENTITY_EXISTS
    assert first.path.read_bytes() == winner_bytes[0]
    survivor = ControlIdentityStore(tmp_path / "control", keys=keys, clock=clock).load()
    assert survivor == winner[0]


def test_renewal_keeps_the_spki_generation_of_the_record_it_replaces(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore, clock: Clock
) -> None:
    store.create()
    fresh = replace_record(store, keys, spki_generation=2)
    clock.now = T0 + timedelta(days=340)
    fresh = ControlIdentityStore(store.path.parent, keys=keys, clock=clock)

    renewed = fresh.renew()

    assert renewed.spki_generation == 2
    assert ControlIdentityStore(store.path.parent, keys=keys).load().spki_generation == 2


def test_renewal_derives_from_the_record_on_disk_when_it_changed_after_the_key_was_read(
    tmp_path: Path,
    store: ControlIdentityStore,
    keys: StateAuthenticationKeyStore,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record replaced between the read and the write is renewed as it is, never reverted to the stale one."""
    store.create()
    replacement_store = ControlIdentityStore(tmp_path / "replacement", keys=keys, clock=clock)
    replacement = replacement_store.create()
    replacement_document = AuthenticatedDocumentStore(
        replacement_store.path.parent,
        DOCUMENT_NAME,
        store_id="control-identity",
        schema_version=1,
        max_bytes=8192,
        keys=keys,
    )
    loaded = replacement_document.load()
    assert loaded is not None
    new_body = {**loaded.record, "spki_generation": 2}
    # The inner versioned document of the store under test: the same instance, so the replacement is "its own write".
    inner = store._document._document
    real_load = keys.load
    fired: list[bool] = []

    def load_then_replace() -> Any:
        key = real_load()
        if not fired:
            fired.append(True)
            tag = key.tag(store._document._message(2, new_body))
            inner.save({"body": new_body, "tag": encode_base64url(tag, exact_length=32)})
        return key

    monkeypatch.setattr(keys, "load", load_then_replace)
    clock.now = T0 + timedelta(days=340)

    renewed = store.renew()
    monkeypatch.undo()

    assert fired
    assert renewed.spki_sha256 == replacement.spki_sha256
    assert renewed.spki_generation == 2
    on_disk = ControlIdentityStore(store.path.parent, keys=keys).load()
    assert on_disk.spki_sha256 == replacement.spki_sha256
    assert on_disk.spki_generation == 2
    assert on_disk.private_key().private_numbers() == replacement.private_key().private_numbers()


def test_renewing_a_missing_identity_writes_nothing(store: ControlIdentityStore) -> None:
    assert fault_of(store.renew) is StoreFault.IDENTITY_MISSING
    assert not store.path.exists()


# --------------------------------------------------------------------------------------------------
# the private key stays inside the identity
# --------------------------------------------------------------------------------------------------


def test_an_identity_cannot_be_exported_by_asdict_astuple_pickle_or_copy(store: ControlIdentityStore) -> None:
    identity = store.create()

    for export in (
        lambda: dataclasses.asdict(identity),
        lambda: dataclasses.astuple(identity),
        lambda: pickle.dumps(identity),
        lambda: copy.copy(identity),
        lambda: copy.deepcopy(identity),
    ):
        with pytest.raises(TypeError):
            export()


def test_neither_repr_nor_str_shows_the_private_scalar(store: ControlIdentityStore) -> None:
    identity = store.create()
    scalar = identity.private_key().private_numbers().private_value.to_bytes(32, "big")

    for rendered in (repr(identity), str(identity), repr(identity.private_key_pkcs8_der) + repr(identity)):
        assert scalar.hex() not in rendered
        assert b64(scalar) not in rendered
        assert str(int.from_bytes(scalar, "big")) not in rendered


def test_an_identity_cannot_be_modified(store: ControlIdentityStore) -> None:
    identity = store.create()
    pin = identity.spki_sha256

    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.spki_sha256 = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.spki_generation = 2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity._scalar = None  # type: ignore[assignment,misc]
    assert not hasattr(identity, "__dict__")  # slotted: no side channel for extra attributes
    assert identity.spki_sha256 == pin


# --------------------------------------------------------------------------------------------------
# what _decode accepts: only a self-signed, SHA-256, end-entity certificate of the stored key
# --------------------------------------------------------------------------------------------------


def build_certificate(
    private_key: ec.EllipticCurvePrivateKey,
    *,
    issuer_name: str = "Remote Energy Meter control",
    algorithm: hashes.SHA256 | hashes.SHA384 | None = None,
    ca: bool | None = False,
    padding: int = 0,
) -> bytes:
    """A certificate for ``private_key`` signed by it, valid in every respect except the one the caller varies."""
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Remote Energy Meter control")])
    issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_name)])
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(7)
        .not_valid_before(T0 - CERTIFICATE_BACKDATE)
        .not_valid_after(T0 + CERTIFICATE_LIFETIME)
    )
    if ca is not None:
        builder = builder.add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    if padding:
        builder = builder.add_extension(
            x509.UnrecognizedExtension(x509.ObjectIdentifier("1.3.6.1.4.1.99999.1"), b"\x00" * padding), critical=False
        )
    certificate = builder.sign(private_key, algorithm or hashes.SHA256())
    return certificate.public_bytes(serialization.Encoding.DER)


def store_with_certificate(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore, certificate_der: bytes
) -> ControlIdentityStore:
    return replace_record(store, keys, certificate_der=b64(certificate_der))


def test_a_hand_built_certificate_of_the_stored_key_is_accepted_as_the_baseline(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore
) -> None:
    identity = store.create()
    fresh = store_with_certificate(store, keys, build_certificate(identity.private_key()))

    assert fresh.load().spki_sha256 == identity.spki_sha256


def test_a_certificate_issued_by_another_name_but_signed_by_the_stored_key_is_inconsistent(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore
) -> None:
    identity = store.create()
    der = build_certificate(identity.private_key(), issuer_name="Some other issuer")

    assert fault_of(store_with_certificate(store, keys, der).load) is StoreFault.IDENTITY_INCONSISTENT


def test_a_certificate_signed_with_a_hash_other_than_sha256_is_inconsistent(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore
) -> None:
    identity = store.create()
    der = build_certificate(identity.private_key(), algorithm=hashes.SHA384())

    assert fault_of(store_with_certificate(store, keys, der).load) is StoreFault.IDENTITY_INCONSISTENT


def test_a_ca_certificate_is_inconsistent(store: ControlIdentityStore, keys: StateAuthenticationKeyStore) -> None:
    identity = store.create()
    der = build_certificate(identity.private_key(), ca=True)

    assert fault_of(store_with_certificate(store, keys, der).load) is StoreFault.IDENTITY_INCONSISTENT


def test_a_certificate_without_basic_constraints_is_inconsistent(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore
) -> None:
    identity = store.create()
    der = build_certificate(identity.private_key(), ca=None)

    assert fault_of(store_with_certificate(store, keys, der).load) is StoreFault.IDENTITY_INCONSISTENT


CURVE_ORDER = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551


@pytest.mark.parametrize("scalar", [CURVE_ORDER, CURVE_ORDER + 1, 2**256 - 1])
def test_a_private_scalar_at_or_above_the_curve_order_is_malformed(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore, scalar: int
) -> None:
    store.create()
    fresh = replace_record(store, keys, private_scalar=b64(scalar.to_bytes(32, "big")))

    assert fault_of(fresh.load) is StoreFault.MALFORMED


def test_a_certificate_over_the_size_bound_is_malformed_even_when_it_is_otherwise_valid(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore
) -> None:
    identity = store.create()
    within = build_certificate(identity.private_key(), padding=3000)
    over = build_certificate(identity.private_key(), padding=4200)
    assert len(within) <= 4096 < len(over)

    assert store_with_certificate(store, keys, within).load().spki_sha256 == identity.spki_sha256
    assert fault_of(store_with_certificate(store, keys, over).load) is StoreFault.MALFORMED


# --------------------------------------------------------------------------------------------------
# time zones
# --------------------------------------------------------------------------------------------------


def test_needs_renewal_refuses_a_naive_datetime(store: ControlIdentityStore) -> None:
    identity = store.create()

    with pytest.raises(ValueError, match="timezone-aware"):
        identity.needs_renewal(datetime(2030, 1, 1, 12, 0, 0))  # noqa: DTZ001 - the naive value is the input under test


def test_a_naive_clock_is_refused_by_create_and_writes_nothing(
    tmp_path: Path, keys: StateAuthenticationKeyStore
) -> None:
    naive = ControlIdentityStore(
        tmp_path / "control",
        keys=keys,
        clock=lambda: datetime(2030, 1, 1, 12, 0, 0),  # noqa: DTZ001
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        naive.create()
    assert not naive.path.exists()


def test_a_naive_clock_is_refused_by_renew_and_keeps_the_record(
    store: ControlIdentityStore, keys: StateAuthenticationKeyStore
) -> None:
    store.create()
    before = store.path.read_bytes()
    naive = ControlIdentityStore(
        store.path.parent,
        keys=keys,
        clock=lambda: datetime(2030, 6, 1, 12, 0, 0),  # noqa: DTZ001
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        naive.renew()
    assert store.path.read_bytes() == before


def test_a_clock_in_another_time_zone_issues_the_same_instant(
    tmp_path: Path, keys: StateAuthenticationKeyStore
) -> None:
    offset = timezone(timedelta(hours=5, minutes=30))
    shifted = ControlIdentityStore(tmp_path / "control", keys=keys, clock=lambda: T0.astimezone(offset))

    identity = shifted.create()

    assert identity.not_before == T0 - CERTIFICATE_BACKDATE
    assert identity.not_after == T0 + CERTIFICATE_LIFETIME


# --------------------------------------------------------------------------------------------------
# the stored record and the module's surface
# --------------------------------------------------------------------------------------------------


def test_the_module_exports_exactly_its_documented_names() -> None:
    assert set(control_identity.__all__) == {
        "CERTIFICATE_BACKDATE",
        "CERTIFICATE_LIFETIME",
        "DOCUMENT_NAME",
        "RENEW_BEFORE_EXPIRY",
        "ControlIdentity",
        "ControlIdentityStore",
    }
    for name in control_identity.__all__:
        assert hasattr(control_identity, name)


def test_the_stored_record_has_exactly_the_documented_keys_and_no_more(store: ControlIdentityStore) -> None:
    identity = store.create()
    envelope = json.loads(store.path.read_bytes())
    record = envelope["record"]

    assert set(record) == {"body", "tag"}
    assert set(record["body"]) == {"certificate_der", "private_scalar", "spki_generation", "spki_sha256"}
    assert record["body"]["spki_sha256"] == identity.spki_sha256
    assert record["body"]["spki_generation"] == 1
