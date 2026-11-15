"""The generated reporter PKI: a ten-year CA, a one-year leaf under it, and a leaf key that renewal never changes."""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import ipaddress
import pickle
import stat
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID

from remote_energy_meter import reporter_pki
from remote_energy_meter.authenticated_store import AuthenticatedDocumentStore, StateAuthenticationKeyStore
from remote_energy_meter.protocol_v1 import encode_base64url
from remote_energy_meter.reporter_pki import (
    CA_DOCUMENT_NAME,
    CA_LIFETIME,
    CERTIFICATE_BACKDATE,
    LEAF_DOCUMENT_NAME,
    LEAF_LIFETIME,
    MAX_DNS_NAMES,
    RENEW_BEFORE_EXPIRY,
    ReporterPkiStore,
)
from remote_energy_meter.secure_store import SecureStoreError, StoreFault

T0 = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)
NAMES = ("ha.example.com",)


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
def pki(tmp_path: Path, keys: StateAuthenticationKeyStore, clock: Clock) -> ReporterPkiStore:
    return ReporterPkiStore(tmp_path / "ca", tmp_path / "leaf", keys=keys, clock=clock)


def fault_of(operation: Callable[[], object]) -> StoreFault:
    with pytest.raises(SecureStoreError) as caught:
        operation()
    return caught.value.fault


def pin_of(certificate_der: bytes) -> str:
    """Recompute a pin independently: unpadded base64url of SHA-256 over the DER SubjectPublicKeyInfo."""
    key = x509.load_der_x509_certificate(certificate_der).public_key()
    spki = key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return base64.urlsafe_b64encode(hashlib.sha256(spki).digest()).rstrip(b"=").decode("ascii")


def b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def rewrite(
    store: ReporterPkiStore, keys: StateAuthenticationKeyStore, *, leaf: bool, **changes: Any
) -> ReporterPkiStore:
    """Re-save a record correctly tagged but altered, and return a fresh store (as after a restart)."""
    document = AuthenticatedDocumentStore(
        (store.leaf_path if leaf else store.ca_path).parent,
        LEAF_DOCUMENT_NAME if leaf else CA_DOCUMENT_NAME,
        store_id="reporter-leaf" if leaf else "reporter-ca",
        schema_version=1,
        max_bytes=16384,
        keys=keys,
    )
    loaded = document.load()
    assert loaded is not None
    document.save({**loaded.record, **changes})
    return ReporterPkiStore(store.ca_path.parent, store.leaf_path.parent, keys=keys)


# --------------------------------------------------------------------------------------------------
# creation
# --------------------------------------------------------------------------------------------------


def test_create_stores_ca_and_leaf_in_separate_owner_only_directories(pki: ReporterPkiStore, tmp_path: Path) -> None:
    pki.create(NAMES)

    for directory in (tmp_path / "ca", tmp_path / "leaf"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(pki.ca_path.stat().st_mode) == stat.S_IMODE(pki.leaf_path.stat().st_mode) == 0o600
    assert [entry.name for entry in (tmp_path / "ca").iterdir()] == [CA_DOCUMENT_NAME]
    assert [entry.name for entry in (tmp_path / "leaf").iterdir()] == [LEAF_DOCUMENT_NAME]


def test_the_leaf_record_never_contains_the_ca_signing_key(pki: ReporterPkiStore) -> None:
    pki.create(NAMES)
    ca_scalar = pki.load_ca().private_key().private_numbers().private_value.to_bytes(32, "big")

    assert encode_base64url(ca_scalar, exact_length=32).encode() not in pki.leaf_path.read_bytes()
    assert ca_scalar not in pki.leaf_path.read_bytes()


def test_the_ca_is_a_ten_year_path_length_zero_ca_with_a_pin(pki: ReporterPkiStore) -> None:
    pki.create(NAMES)
    ca = pki.load_ca()
    certificate = x509.load_der_x509_certificate(ca.certificate_der)

    certificate.verify_directly_issued_by(certificate)
    constraints = certificate.extensions.get_extension_for_class(x509.BasicConstraints)
    assert constraints.critical and constraints.value.ca is True and constraints.value.path_length == 0
    usage = certificate.extensions.get_extension_for_class(x509.KeyUsage).value
    assert usage.key_cert_sign and usage.crl_sign and not usage.digital_signature
    assert ca.not_before == T0 - CERTIFICATE_BACKDATE
    assert ca.not_after == T0 + CA_LIFETIME == T0 + timedelta(days=3650)
    assert ca.spki_sha256 == pin_of(ca.certificate_der)
    assert ca.ca_generation == 1


def test_the_leaf_is_a_one_year_server_certificate_issued_by_the_ca(pki: ReporterPkiStore) -> None:
    leaf = pki.create(("b.example.com", "a.example.com"))
    ca = pki.load_ca()
    certificate = x509.load_der_x509_certificate(leaf.certificate_der)

    certificate.verify_directly_issued_by(x509.load_der_x509_certificate(ca.certificate_der))
    assert leaf.not_before == T0 - CERTIFICATE_BACKDATE
    assert leaf.not_after == T0 + LEAF_LIFETIME == T0 + timedelta(days=365)
    assert leaf.dns_names == ("a.example.com", "b.example.com")
    san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert sorted(san.get_values_for_type(x509.DNSName)) == ["a.example.com", "b.example.com"]
    assert list(certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value) == [
        ExtendedKeyUsageOID.SERVER_AUTH
    ]
    constraints = certificate.extensions.get_extension_for_class(x509.BasicConstraints).value
    assert constraints.ca is False
    assert leaf.spki_sha256 == pin_of(leaf.certificate_der) != ca.spki_sha256
    assert leaf.ca_spki_sha256 == ca.spki_sha256
    assert leaf.ca_generation == ca.ca_generation


def test_a_fresh_instance_reloads_identical_ca_and_leaf(
    pki: ReporterPkiStore, tmp_path: Path, keys: StateAuthenticationKeyStore, clock: Clock
) -> None:
    created = pki.create(NAMES)
    restarted = ReporterPkiStore(tmp_path / "ca", tmp_path / "leaf", keys=keys, clock=clock)

    assert restarted.load_leaf() == created
    assert restarted.load_ca().spki_sha256 == created.ca_spki_sha256
    assert restarted.load_leaf().private_key().private_numbers() == created.private_key().private_numbers()


def test_an_existing_ca_is_never_replaced(pki: ReporterPkiStore) -> None:
    pki.create(NAMES)
    before = pki.ca_path.read_bytes()

    assert fault_of(lambda: pki.create(NAMES)) is StoreFault.IDENTITY_EXISTS
    assert fault_of(pki.create_ca) is StoreFault.IDENTITY_EXISTS
    assert pki.ca_path.read_bytes() == before


def test_an_existing_leaf_is_never_replaced_by_create_leaf(pki: ReporterPkiStore) -> None:
    pki.create(NAMES)
    before = pki.leaf_path.read_bytes()

    assert fault_of(lambda: pki.create_leaf(NAMES)) is StoreFault.IDENTITY_EXISTS
    assert pki.leaf_path.read_bytes() == before


def test_a_creation_interrupted_after_the_ca_is_finished_by_create_leaf(pki: ReporterPkiStore) -> None:
    ca = pki.create_ca()

    assert fault_of(pki.load_leaf) is StoreFault.IDENTITY_MISSING
    leaf = pki.create_leaf(NAMES)

    assert pki.load_leaf() == leaf
    assert leaf.ca_spki_sha256 == ca.spki_sha256


def test_a_missing_ca_is_reported_and_nothing_is_created(pki: ReporterPkiStore) -> None:
    assert fault_of(pki.load_ca) is StoreFault.IDENTITY_MISSING
    assert fault_of(pki.load_leaf) is StoreFault.IDENTITY_MISSING
    assert fault_of(lambda: pki.create_leaf(NAMES)) is StoreFault.IDENTITY_MISSING
    assert fault_of(pki.renew_leaf) is StoreFault.IDENTITY_MISSING
    assert not pki.ca_path.exists() and not pki.leaf_path.exists()


def test_creating_without_a_state_authentication_key_fails_closed(tmp_path: Path, clock: Clock) -> None:
    orphan = ReporterPkiStore(
        tmp_path / "ca", tmp_path / "leaf", keys=StateAuthenticationKeyStore(tmp_path / "none"), clock=clock
    )

    assert fault_of(lambda: orphan.create(NAMES)) is StoreFault.KEY_MISSING
    assert not orphan.ca_path.exists()


def test_the_ca_and_leaf_directories_must_differ(tmp_path: Path, keys: StateAuthenticationKeyStore) -> None:
    with pytest.raises(ValueError, match="different directories"):
        ReporterPkiStore(tmp_path / "same", tmp_path / "same", keys=keys)


def test_private_keys_never_appear_in_repr_pickle_copy_or_asdict(pki: ReporterPkiStore) -> None:
    leaf = pki.create(NAMES)
    ca = pki.load_ca()
    for item in (leaf, ca):
        scalar = item.private_key().private_numbers().private_value
        assert str(scalar) not in repr(item)
        assert scalar.to_bytes(32, "big").hex() not in repr(item)
        assert "certificate_der" not in repr(item)
        for operation in (pickle.dumps, copy.copy, copy.deepcopy, dataclasses.asdict, dataclasses.astuple):
            with pytest.raises(TypeError):
                operation(item)


def test_the_leaf_key_exports_as_pkcs8_for_the_edge(pki: ReporterPkiStore) -> None:
    leaf = pki.create(NAMES)

    loaded = serialization.load_der_private_key(leaf.private_key_pkcs8_der(), password=None)

    assert isinstance(loaded, ec.EllipticCurvePrivateKey)
    assert loaded.private_numbers() == leaf.private_key().private_numbers()


# --------------------------------------------------------------------------------------------------
# host names
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [
        [],
        ["Upper.example.com"],
        ["ha.example.com."],
        ["-bad.example.com"],
        ["bad-.example.com"],
        ["under_score.example.com"],
        ["192.0.2.1"],
        ["1"],
        ["ha.example.com", "ha.example.com"],
        [""],
        ["a" * 64 + ".example.com"],
        ["a." * 130 + "com"],
        ["é.example.com"],
        ["ha.example.com"] * (MAX_DNS_NAMES + 1),
        "ha.example.com",
    ],
)
def test_invalid_host_names_are_rejected_before_anything_is_written(pki: ReporterPkiStore, names: Any) -> None:
    with pytest.raises(ValueError):
        pki.create(names)
    assert not pki.ca_path.exists()


def test_the_maximum_number_of_names_is_accepted(pki: ReporterPkiStore) -> None:
    names = [f"h{index}.example.com" for index in range(MAX_DNS_NAMES)]

    assert len(pki.create(names).dns_names) == MAX_DNS_NAMES


# --------------------------------------------------------------------------------------------------
# renewal
# --------------------------------------------------------------------------------------------------


def test_renewal_keeps_the_leaf_key_and_spki_and_issues_a_new_certificate(pki: ReporterPkiStore, clock: Clock) -> None:
    created = pki.create(NAMES)
    ca_before = pki.load_ca()
    clock.now = T0 + timedelta(days=340)

    renewed = pki.renew_leaf()

    assert renewed.spki_sha256 == created.spki_sha256 == pin_of(renewed.certificate_der)
    assert renewed.private_key().private_numbers() == created.private_key().private_numbers()
    assert renewed.certificate_der != created.certificate_der
    assert renewed.not_after == clock.now + LEAF_LIFETIME
    assert renewed.dns_names == created.dns_names
    assert pki.load_leaf() == renewed
    assert pki.load_ca() == ca_before  # the CA is untouched
    x509.load_der_x509_certificate(renewed.certificate_der).verify_directly_issued_by(
        x509.load_der_x509_certificate(ca_before.certificate_der)
    )


def test_renewal_can_change_host_names_under_the_same_key(pki: ReporterPkiStore) -> None:
    created = pki.create(NAMES)

    renewed = pki.renew_leaf(["new.example.com"])

    assert renewed.dns_names == ("new.example.com",)
    assert renewed.spki_sha256 == created.spki_sha256


def test_invalid_names_on_renewal_raise_value_error_and_change_nothing(pki: ReporterPkiStore) -> None:
    pki.create(NAMES)
    before = pki.leaf_path.read_bytes()

    with pytest.raises(ValueError):
        pki.renew_leaf(["Bad.example.com"])

    assert pki.leaf_path.read_bytes() == before


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (timedelta(0), False),
        (LEAF_LIFETIME - RENEW_BEFORE_EXPIRY - timedelta(seconds=1), False),
        (LEAF_LIFETIME - RENEW_BEFORE_EXPIRY, True),
        (LEAF_LIFETIME, True),
    ],
)
def test_the_leaf_needs_renewal_exactly_thirty_days_before_expiry(
    pki: ReporterPkiStore, elapsed: timedelta, expected: bool
) -> None:
    leaf = pki.create(NAMES)

    assert leaf.needs_renewal(T0 + elapsed) is expected
    with pytest.raises(ValueError, match="timezone-aware"):
        leaf.needs_renewal(datetime(2030, 1, 1))  # noqa: DTZ001


def test_many_renewals_never_change_the_leaf_or_ca_pins(pki: ReporterPkiStore, clock: Clock) -> None:
    leaf = pki.create(NAMES)
    ca_pin = leaf.ca_spki_sha256
    for month in range(1, 8):
        clock.now = T0 + timedelta(days=30 * month)
        renewed = pki.renew_leaf()
        assert (renewed.spki_sha256, renewed.ca_spki_sha256) == (leaf.spki_sha256, ca_pin)


def test_a_leaf_never_outlives_its_ca(pki: ReporterPkiStore, clock: Clock) -> None:
    pki.create(NAMES)
    ca = pki.load_ca()
    clock.now = ca.not_after - timedelta(days=100)

    leaf = pki.renew_leaf()

    assert leaf.not_after == ca.not_after
    assert leaf.needs_renewal(clock.now) is False  # a hundred days remain, more than the renewal window


def test_nothing_is_issued_under_an_expired_ca(pki: ReporterPkiStore, clock: Clock) -> None:
    pki.create(NAMES)
    before = pki.leaf_path.read_bytes()
    clock.now = pki.load_ca().not_after

    assert fault_of(pki.renew_leaf) is StoreFault.CERTIFICATE_EXPIRED
    assert pki.leaf_path.read_bytes() == before


def test_a_naive_clock_is_rejected(tmp_path: Path, keys: StateAuthenticationKeyStore) -> None:
    naive = ReporterPkiStore(tmp_path / "ca", tmp_path / "leaf", keys=keys, clock=lambda: datetime(2030, 1, 1))  # noqa: DTZ001

    with pytest.raises(ValueError, match="timezone-aware"):
        naive.create(NAMES)


# --------------------------------------------------------------------------------------------------
# tampering and inconsistency
# --------------------------------------------------------------------------------------------------


def test_an_edited_record_fails_authentication(pki: ReporterPkiStore, keys: StateAuthenticationKeyStore) -> None:
    pki.create(NAMES)
    pki.ca_path.write_bytes(pki.ca_path.read_bytes().replace(b'"ca_generation":1', b'"ca_generation":2'))

    assert fault_of(ReporterPkiStore(pki.ca_path.parent, pki.leaf_path.parent, keys=keys).load_ca) is (
        StoreFault.AUTHENTICATION_FAILED
    )


def test_a_leaf_from_another_pki_is_rejected(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore, tmp_path: Path, clock: Clock
) -> None:
    pki.create(NAMES)
    other = ReporterPkiStore(tmp_path / "other-ca", tmp_path / "other-leaf", keys=keys, clock=clock)
    stranger = other.create(NAMES)
    stolen = {
        "certificate_der": b64(stranger.certificate_der),
        "private_scalar": encode_base64url(
            stranger.private_key().private_numbers().private_value.to_bytes(32, "big"), exact_length=32
        ),
        "spki_sha256": stranger.spki_sha256,
    }

    fresh = rewrite(pki, keys, leaf=True, **stolen)

    assert fault_of(fresh.load_leaf) is StoreFault.IDENTITY_INCONSISTENT


def test_a_leaf_naming_another_ca_pin_or_generation_is_rejected(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore
) -> None:
    pki.create(NAMES)
    wrong_pin = rewrite(pki, keys, leaf=True, ca_spki_sha256=encode_base64url(b"\x01" * 32, exact_length=32))
    assert fault_of(wrong_pin.load_leaf) is StoreFault.IDENTITY_INCONSISTENT

    wrong_generation = rewrite(pki, keys, leaf=True, ca_generation=2)
    assert fault_of(wrong_generation.load_leaf) is StoreFault.IDENTITY_INCONSISTENT


def test_a_swapped_ca_with_a_different_key_breaks_the_leaf_chain(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore, tmp_path: Path, clock: Clock
) -> None:
    pki.create(NAMES)
    impostor = ReporterPkiStore(tmp_path / "x-ca", tmp_path / "x-leaf", keys=keys, clock=clock)
    fake_ca = impostor.create_ca()
    fresh = rewrite(
        pki,
        keys,
        leaf=False,
        certificate_der=b64(fake_ca.certificate_der),
        private_scalar=encode_base64url(
            fake_ca.private_key().private_numbers().private_value.to_bytes(32, "big"), exact_length=32
        ),
        spki_sha256=fake_ca.spki_sha256,
    )

    # The impostor CA is itself authentic and self-consistent; it is the leaf that no longer chains to it.
    assert fresh.load_ca().spki_sha256 == fake_ca.spki_sha256
    assert fault_of(fresh.load_leaf) is StoreFault.IDENTITY_INCONSISTENT


def test_a_ca_whose_pin_is_not_the_hash_of_its_key_is_inconsistent(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore
) -> None:
    pki.create(NAMES)
    fresh = rewrite(pki, keys, leaf=False, spki_sha256=encode_base64url(b"\x02" * 32, exact_length=32))

    assert fault_of(fresh.load_ca) is StoreFault.IDENTITY_INCONSISTENT


def test_a_ca_private_key_that_does_not_match_its_certificate_is_inconsistent(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore
) -> None:
    pki.create(NAMES)
    other = ec.generate_private_key(ec.SECP256R1()).private_numbers().private_value.to_bytes(32, "big")
    fresh = rewrite(pki, keys, leaf=False, private_scalar=encode_base64url(other, exact_length=32))

    assert fault_of(fresh.load_ca) is StoreFault.IDENTITY_INCONSISTENT


def test_a_leaf_presented_as_the_ca_certificate_is_inconsistent(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore
) -> None:
    leaf = pki.create(NAMES)
    leaf_scalar = leaf.private_key().private_numbers().private_value.to_bytes(32, "big")
    fresh = rewrite(
        pki,
        keys,
        leaf=False,
        certificate_der=b64(leaf.certificate_der),
        private_scalar=encode_base64url(leaf_scalar, exact_length=32),
        spki_sha256=leaf.spki_sha256,
    )

    assert fault_of(fresh.load_ca) is StoreFault.IDENTITY_INCONSISTENT


def test_a_ca_certificate_with_a_corrupted_signature_is_inconsistent(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore
) -> None:
    pki.create(NAMES)
    der = bytearray(pki.load_ca().certificate_der)
    der[-1] ^= 0x01
    fresh = rewrite(pki, keys, leaf=False, certificate_der=b64(bytes(der)))

    assert fault_of(fresh.load_ca) is StoreFault.IDENTITY_INCONSISTENT


def test_a_leaf_certificate_with_a_corrupted_signature_is_inconsistent(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore
) -> None:
    leaf = pki.create(NAMES)
    der = bytearray(leaf.certificate_der)
    der[-1] ^= 0x01
    fresh = rewrite(pki, keys, leaf=True, certificate_der=b64(bytes(der)))

    assert fault_of(fresh.load_leaf) is StoreFault.IDENTITY_INCONSISTENT


def test_a_leaf_whose_listed_names_differ_from_its_certificate_is_inconsistent(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore
) -> None:
    pki.create(NAMES)
    fresh = rewrite(pki, keys, leaf=True, dns_names=["other.example.com"])

    assert fault_of(fresh.load_leaf) is StoreFault.IDENTITY_INCONSISTENT


@pytest.mark.parametrize("names", [["b.example.com", "a.example.com"], ["Bad.example.com"], [], "ha.example.com", [1]])
def test_a_leaf_with_malformed_or_non_canonical_names_is_malformed(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore, names: Any
) -> None:
    pki.create(["a.example.com", "b.example.com"])
    fresh = rewrite(pki, keys, leaf=True, dns_names=names)

    assert fault_of(fresh.load_leaf) is StoreFault.MALFORMED


@pytest.mark.parametrize(
    ("leaf", "changes"),
    [
        (False, {"ca_generation": 0}),
        (False, {"ca_generation": "1"}),
        (False, {"private_scalar": "short"}),
        (False, {"private_scalar": encode_base64url(b"\x00" * 32, exact_length=32)}),
        (False, {"spki_sha256": "nope"}),
        (False, {"certificate_der": "!!!"}),
        (False, {"certificate_der": "AAAA"}),
        (True, {"ca_generation": 0}),
        (True, {"ca_spki_sha256": "nope"}),
        (True, {"spki_sha256": 1}),
        (True, {"certificate_der": "!!!"}),
        (True, {"private_scalar": "short"}),
    ],
)
def test_undecodable_or_mistyped_fields_are_malformed(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore, leaf: bool, changes: dict[str, Any]
) -> None:
    pki.create(NAMES)
    fresh = rewrite(pki, keys, leaf=leaf, **changes)

    assert fault_of(fresh.load_leaf if leaf else fresh.load_ca) is StoreFault.MALFORMED


def test_records_with_missing_or_extra_fields_are_malformed(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore
) -> None:
    pki.create(NAMES)
    ca_document = AuthenticatedDocumentStore(
        pki.ca_path.parent, CA_DOCUMENT_NAME, store_id="reporter-ca", schema_version=1, max_bytes=16384, keys=keys
    )
    loaded = ca_document.load()
    assert loaded is not None
    ca_document.save({**loaded.record, "extra": 1})
    assert fault_of(ReporterPkiStore(pki.ca_path.parent, pki.leaf_path.parent, keys=keys).load_ca) is (
        StoreFault.MALFORMED
    )


def test_the_module_exports_its_public_names() -> None:
    for name in reporter_pki.__all__:
        assert hasattr(reporter_pki, name)


def test_a_ca_created_by_another_writer_meanwhile_is_reported_as_existing_and_kept(
    pki: ReporterPkiStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    keys: StateAuthenticationKeyStore,
    clock: Clock,
) -> None:
    """The pre-check sees no CA, another writer then creates one: the exclusive create replaces nothing."""
    rival = ReporterPkiStore(tmp_path / "ca", tmp_path / "rival-leaf", keys=keys, clock=clock)
    rival.create_ca()
    winner = pki.ca_path.read_bytes()
    monkeypatch.setattr(type(pki._ca_document), "load", lambda self: None)

    assert fault_of(pki.create_ca) is StoreFault.IDENTITY_EXISTS
    assert pki.ca_path.read_bytes() == winner


def test_a_leaf_created_by_another_writer_meanwhile_is_reported_as_existing_and_kept(
    pki: ReporterPkiStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    keys: StateAuthenticationKeyStore,
    clock: Clock,
) -> None:
    pki.create(NAMES)
    winner = pki.leaf_path.read_bytes()
    real_load = type(pki._leaf_document).load

    def load_leaf_as_absent(self: AuthenticatedDocumentStore) -> Any:
        return None if self.path.name == LEAF_DOCUMENT_NAME else real_load(self)

    monkeypatch.setattr(AuthenticatedDocumentStore, "load", load_leaf_as_absent)

    assert fault_of(lambda: pki.create_leaf(NAMES)) is StoreFault.IDENTITY_EXISTS
    assert pki.leaf_path.read_bytes() == winner


def test_renewing_when_the_leaf_vanishes_under_the_lock_reports_it_missing(
    pki: ReporterPkiStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    pki.create(NAMES)
    leaf_document = pki._leaf_document
    real_update = leaf_document.update

    def update_with_nothing(build: Callable[[Any], Any]) -> Any:
        return real_update(lambda _current: build(None))

    monkeypatch.setattr(leaf_document, "update", update_with_nothing)

    assert fault_of(pki.renew_leaf) is StoreFault.IDENTITY_MISSING


# --------------------------------------------------------------------------------------------------
# load-time certificate checks (authentic records whose certificates are wrongly shaped)
# --------------------------------------------------------------------------------------------------


def key_usage(**enabled: bool) -> x509.KeyUsage:
    flags = dict.fromkeys(
        (
            "digital_signature",
            "content_commitment",
            "key_encipherment",
            "data_encipherment",
            "key_agreement",
            "key_cert_sign",
            "crl_sign",
            "encipher_only",
            "decipher_only",
        ),
        False,
    )
    flags.update(enabled)
    return x509.KeyUsage(**flags)


_CA_BASIC = x509.BasicConstraints(ca=True, path_length=0)
_CA_USAGE = key_usage(key_cert_sign=True, crl_sign=True)
_LEAF_BASIC = x509.BasicConstraints(ca=False, path_length=None)
_LEAF_USAGE = key_usage(digital_signature=True)
_LEAF_EKU = (ExtendedKeyUsageOID.SERVER_AUTH,)


def forge_ca_certificate(
    pki: ReporterPkiStore,
    *,
    basic: x509.BasicConstraints | None = _CA_BASIC,
    basic_critical: bool = True,
    usage: x509.KeyUsage | None = _CA_USAGE,
) -> str:
    """A self-signed certificate over the real CA key, shaped as told; ``None`` omits the extension."""
    ca = pki.load_ca()
    key = ca.private_key()
    name = x509.load_der_x509_certificate(ca.certificate_der).subject
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(ca.not_before)
        .not_valid_after(ca.not_after)
    )
    if basic is not None:
        builder = builder.add_extension(basic, critical=basic_critical)
    if usage is not None:
        builder = builder.add_extension(usage, critical=True)
    return b64(builder.sign(key, hashes.SHA256()).public_bytes(serialization.Encoding.DER))


def forge_leaf_certificate(
    pki: ReporterPkiStore,
    *,
    basic: x509.BasicConstraints | None = _LEAF_BASIC,
    basic_critical: bool = True,
    usage: x509.KeyUsage | None = _LEAF_USAGE,
    eku: Sequence[x509.ObjectIdentifier] | None = _LEAF_EKU,
    extra_names: list[x509.GeneralName] | None = None,
    not_before_shift: timedelta | None = None,
    not_after_shift: timedelta | None = None,
) -> str:
    """A certificate over the real leaf key signed by the real CA key, shaped as told; ``None`` omits an extension.

    The validity is the leaf's own unless a shift relative to the CA's validity is given.
    """
    ca = pki.load_ca()
    leaf = pki.load_leaf()
    ca_certificate = x509.load_der_x509_certificate(ca.certificate_der)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.load_der_x509_certificate(leaf.certificate_der).subject)
        .issuer_name(ca_certificate.subject)
        .public_key(leaf.private_key().public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(leaf.not_before if not_before_shift is None else ca.not_before + not_before_shift)
        .not_valid_after(leaf.not_after if not_after_shift is None else ca.not_after + not_after_shift)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(n) for n in leaf.dns_names] + (extra_names or [])),
            critical=False,
        )
    )
    if basic is not None:
        builder = builder.add_extension(basic, critical=basic_critical)
    if usage is not None:
        builder = builder.add_extension(usage, critical=True)
    if eku is not None:
        builder = builder.add_extension(x509.ExtendedKeyUsage(eku), critical=False)
    return b64(builder.sign(ca.private_key(), hashes.SHA256()).public_bytes(serialization.Encoding.DER))


def test_a_forged_ca_certificate_with_the_issued_shape_loads(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore
) -> None:
    """Guards the helper: the failures below are caused by the one deviation, not by the forgery itself."""
    created = pki.create(NAMES)

    fresh = rewrite(pki, keys, leaf=False, certificate_der=forge_ca_certificate(pki))

    assert fresh.load_ca().spki_sha256 == created.ca_spki_sha256


def test_a_forged_leaf_certificate_with_the_issued_shape_loads(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore
) -> None:
    """Guards the helper: the failures below are caused by the one deviation, not by the forgery itself."""
    pki.create(NAMES)

    fresh = rewrite(pki, keys, leaf=True, certificate_der=forge_leaf_certificate(pki))

    assert fresh.load_leaf().dns_names == NAMES


@pytest.mark.parametrize(
    "deviation",
    [
        pytest.param({"basic": x509.BasicConstraints(ca=True, path_length=None)}, id="path-length-unlimited"),
        pytest.param({"basic": x509.BasicConstraints(ca=True, path_length=1)}, id="path-length-one"),
        pytest.param({"basic_critical": False}, id="basic-constraints-not-critical"),
        pytest.param({"usage": None}, id="no-key-usage"),
        pytest.param({"usage": key_usage(crl_sign=True)}, id="key-usage-without-key-cert-sign"),
    ],
)
def test_a_ca_certificate_that_is_not_a_critical_path_length_zero_signing_ca_is_inconsistent(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore, deviation: dict[str, Any]
) -> None:
    pki.create(NAMES)
    fresh = rewrite(pki, keys, leaf=False, certificate_der=forge_ca_certificate(pki, **deviation))

    assert fault_of(fresh.load_ca) is StoreFault.IDENTITY_INCONSISTENT


_SERVER = ExtendedKeyUsageOID.SERVER_AUTH
_CLIENT = ExtendedKeyUsageOID.CLIENT_AUTH
_DAY = timedelta(days=1)
_MINUTE = timedelta(minutes=1)


@pytest.mark.parametrize(
    "deviation",
    [
        pytest.param({"eku": [_CLIENT]}, id="client-auth-only"),
        pytest.param({"eku": None}, id="no-extended-key-usage"),
        pytest.param({"eku": [_SERVER, _CLIENT]}, id="server-auth-plus-another"),
        pytest.param({"usage": None}, id="no-key-usage"),
        pytest.param({"usage": key_usage(digital_signature=True, key_cert_sign=True)}, id="key-cert-sign"),
        pytest.param({"usage": key_usage(key_agreement=True)}, id="no-digital-signature"),
        pytest.param({"basic": None}, id="no-basic-constraints"),
        pytest.param({"basic_critical": False}, id="basic-constraints-not-critical"),
        pytest.param({"extra_names": [x509.IPAddress(ipaddress.ip_address("192.0.2.1"))]}, id="extra-ip-address-name"),
        pytest.param({"extra_names": [x509.UniformResourceIdentifier("https://example.com/")]}, id="extra-uri-name"),
        pytest.param({"not_after_shift": _DAY}, id="outlives-the-ca"),
        pytest.param({"not_before_shift": -_MINUTE}, id="starts-before-the-ca"),
    ],
)
def test_a_leaf_certificate_that_is_not_a_plain_server_authentication_leaf_is_inconsistent(
    pki: ReporterPkiStore, keys: StateAuthenticationKeyStore, deviation: dict[str, Any]
) -> None:
    pki.create(NAMES)
    fresh = rewrite(pki, keys, leaf=True, certificate_der=forge_leaf_certificate(pki, **deviation))

    assert fault_of(fresh.load_leaf) is StoreFault.IDENTITY_INCONSISTENT


# --------------------------------------------------------------------------------------------------
# issuance under a clock that stepped back
# --------------------------------------------------------------------------------------------------


def test_a_leaf_issued_after_the_clock_stepped_back_never_starts_before_the_ca(
    pki: ReporterPkiStore, clock: Clock
) -> None:
    ca = pki.create_ca()
    clock.now = T0 - timedelta(days=1)

    leaf = pki.create_leaf(NAMES)

    assert leaf.not_before >= ca.not_before
    assert pki.load_leaf() == leaf


def test_a_renewed_leaf_after_the_clock_stepped_back_never_starts_before_the_ca(
    pki: ReporterPkiStore, clock: Clock
) -> None:
    ca = pki.create_ca()
    pki.create_leaf(NAMES)
    clock.now = T0 - timedelta(days=1)

    renewed = pki.renew_leaf()

    assert renewed.not_before >= ca.not_before


def test_nothing_is_issued_at_the_first_instant_a_ca_is_valid_when_the_backdate_would_precede_it(
    pki: ReporterPkiStore, clock: Clock
) -> None:
    ca = pki.create_ca()
    clock.now = ca.not_before + timedelta(minutes=1)  # inside the backdate window: now - 5 min precedes the CA

    leaf = pki.create_leaf(NAMES)

    assert leaf.not_before >= ca.not_before


# --------------------------------------------------------------------------------------------------
# directories
# --------------------------------------------------------------------------------------------------


def test_directories_that_are_the_same_through_a_symlink_are_rejected(
    tmp_path: Path, keys: StateAuthenticationKeyStore
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="different directories"):
        ReporterPkiStore(real, alias, keys=keys)
    with pytest.raises(ValueError, match="different directories"):
        ReporterPkiStore(alias, real, keys=keys)


def test_directories_that_are_the_same_through_dot_dot_spelling_are_rejected(
    tmp_path: Path, keys: StateAuthenticationKeyStore
) -> None:
    with pytest.raises(ValueError, match="different directories"):
        ReporterPkiStore(tmp_path / "d", tmp_path / "x" / ".." / "d", keys=keys)


def test_a_directory_nested_inside_the_other_is_rejected_either_way_round(
    tmp_path: Path, keys: StateAuthenticationKeyStore
) -> None:
    with pytest.raises(ValueError):
        ReporterPkiStore(tmp_path / "outer", tmp_path / "outer" / "inner", keys=keys)
    with pytest.raises(ValueError):
        ReporterPkiStore(tmp_path / "outer" / "inner", tmp_path / "outer", keys=keys)


def test_a_nested_directory_through_a_symlink_is_rejected(tmp_path: Path, keys: StateAuthenticationKeyStore) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outer, target_is_directory=True)

    with pytest.raises(ValueError):
        ReporterPkiStore(outer, alias / "inner", keys=keys)


def test_siblings_with_a_common_name_prefix_are_not_treated_as_nested(
    tmp_path: Path, keys: StateAuthenticationKeyStore
) -> None:
    ReporterPkiStore(tmp_path / "ca", tmp_path / "ca-leaf", keys=keys)


def test_a_failed_leaf_directory_leaves_no_ca_behind(
    tmp_path: Path, keys: StateAuthenticationKeyStore, clock: Clock
) -> None:
    blocker = tmp_path / "leaf"
    blocker.write_text("not a directory")
    store = ReporterPkiStore(tmp_path / "ca", blocker, keys=keys, clock=clock)

    with pytest.raises(SecureStoreError):
        store.create(NAMES)

    assert not store.ca_path.exists()


# --------------------------------------------------------------------------------------------------
# host names, stricter forms
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [
        ["a.0x7f"],
        ["1.2.3.4"],
        ["host.2lan"],
        ["xn--zz"],
        [b"ha.example.com"],
        [1],
        [None],
        b"ha",
        bytearray(b"ha"),
        [bytearray(b"ha")],
    ],
)
def test_ip_like_invalid_punycode_and_non_text_names_are_rejected_before_anything_is_written(
    pki: ReporterPkiStore, names: Any
) -> None:
    with pytest.raises(ValueError):
        pki.create(names)
    assert not pki.ca_path.exists()


@pytest.mark.parametrize("name", ["xn--bcher-kva.example", "server-a.example.com"])
def test_valid_punycode_and_ordinary_names_are_accepted(pki: ReporterPkiStore, name: str) -> None:
    assert pki.create([name]).dns_names == (name,)


# --------------------------------------------------------------------------------------------------
# expiry on first issuance
# --------------------------------------------------------------------------------------------------


def test_no_first_leaf_is_issued_under_an_expired_ca(pki: ReporterPkiStore, clock: Clock) -> None:
    pki.create_ca()
    clock.now = pki.load_ca().not_after

    assert fault_of(lambda: pki.create_leaf(NAMES)) is StoreFault.CERTIFICATE_EXPIRED
    assert not pki.leaf_path.exists()
