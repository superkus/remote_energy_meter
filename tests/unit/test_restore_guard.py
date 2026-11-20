"""Restore checks: the one-sided comparator, and the durable post-restore quarantine with its fail-closed gate."""

from __future__ import annotations

import stat
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from remote_energy_meter import restore_guard
from remote_energy_meter.authenticated_store import AuthenticatedDocumentStore, StateAuthenticationKeyStore
from remote_energy_meter.control_authority import AuthorizedControl, ControlAuthority
from remote_energy_meter.protocol_v1 import encode_base64url
from remote_energy_meter.restore_guard import (
    DOCUMENT_NAME,
    MAX_RESTORED_REPORTERS,
    ControlComparison,
    PeerControlState,
    Quarantine,
    QuarantineRefusal,
    QuarantineRefused,
    QuarantineStore,
    compare_control_state,
    evaluate_gate,
)
from remote_energy_meter.secure_store import SecureStoreError, StoreFault

T0 = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)
PIN_A = encode_base64url(b"\x11" * 32, exact_length=32)
PIN_B = encode_base64url(b"\x22" * 32, exact_length=32)
SPKI_A = encode_base64url(b"\x33" * 32, exact_length=32)
SPKI_B = encode_base64url(b"\x44" * 32, exact_length=32)


def reporter(n: int) -> str:
    return encode_base64url(bytes([n]) * 32, exact_length=32)


# --------------------------------------------------------------------------------------------------
# one-sided restore
# --------------------------------------------------------------------------------------------------


def local_state(**overrides: Any) -> ControlAuthority:
    authority = AuthorizedControl(
        jwk=b"{}", thumbprint=PIN_A, key_generation=3, spki_sha256=SPKI_A, spki_generation=2, authorized_at=1
    )
    values: dict[str, Any] = {
        "authority": authority,
        "key_high_water": 3,
        "key_revoked_through": 0,
        "spki_revoked_through": 0,
        "recovery_high_water": 1,
        "consumed": (),
        "results": (),
    }
    values.update(overrides)
    return ControlAuthority(**values)


def peer(**overrides: Any) -> PeerControlState:
    values: dict[str, Any] = {
        "key_thumbprint": PIN_A,
        "key_generation": 3,
        "spki_sha256": SPKI_A,
        "spki_generation": 2,
        "recovery_generation": 1,
    }
    values.update(overrides)
    return PeerControlState(**values)


def test_identical_state_matches_and_is_the_only_outcome_that_enables_control() -> None:
    outcome = compare_control_state(local_state(), peer())

    assert outcome is ControlComparison.MATCH
    assert [member for member in ControlComparison if member.control_enabled] == [ControlComparison.MATCH]


@pytest.mark.parametrize(
    "overrides",
    [
        {"key_generation": 4},
        {"spki_generation": 3},
        {"recovery_generation": 2},
        {"key_generation": 9, "spki_generation": 9},
    ],
)
def test_a_peer_ahead_on_any_generation_means_the_add_on_was_rolled_back(overrides: dict[str, Any]) -> None:
    assert compare_control_state(local_state(), peer(**overrides)) is ControlComparison.LOCAL_BEHIND


@pytest.mark.parametrize(
    "overrides",
    [
        {"key_generation": 2},
        {"spki_generation": 1},
        {"recovery_generation": 0},
        {"key_generation": 1, "spki_generation": 1, "recovery_generation": 0},
    ],
)
def test_a_peer_behind_on_any_generation_means_the_integration_was_rolled_back(overrides: dict[str, Any]) -> None:
    assert compare_control_state(local_state(), peer(**overrides)) is ControlComparison.PEER_BEHIND


@pytest.mark.parametrize(
    "overrides",
    [
        {"key_generation": 4, "spki_generation": 1},
        {"key_generation": 2, "spki_generation": 3},
        {"spki_generation": 3, "recovery_generation": 0},
        {"key_generation": 4, "recovery_generation": 0},
    ],
)
def test_generations_that_disagree_in_opposite_directions_are_inconsistent(overrides: dict[str, Any]) -> None:
    assert compare_control_state(local_state(), peer(**overrides)) is ControlComparison.INCONSISTENT


@pytest.mark.parametrize(
    "overrides", [{"key_thumbprint": PIN_B}, {"spki_sha256": SPKI_B}, {"key_thumbprint": PIN_B, "spki_sha256": SPKI_B}]
)
def test_equal_generations_with_different_pins_mean_a_substituted_key_or_spki(overrides: dict[str, Any]) -> None:
    assert compare_control_state(local_state(), peer(**overrides)) is ControlComparison.SUBSTITUTED


@pytest.mark.parametrize(
    ("local", "overrides"),
    [
        (local_state(key_revoked_through=2, authority=None), {"key_generation": 2}),
        (local_state(key_revoked_through=2), {"key_generation": 1}),
        (local_state(spki_revoked_through=1), {"spki_generation": 1}),
        (local_state(key_revoked_through=3, authority=None), {"key_generation": 3}),
    ],
)
def test_a_peer_holding_a_generation_the_add_on_revoked_is_flagged(
    local: ControlAuthority, overrides: dict[str, Any]
) -> None:
    assert compare_control_state(local, peer(**overrides)) is ControlComparison.PEER_REVOKED


def test_a_peer_that_never_paired_is_not_mistaken_for_a_revoked_one() -> None:
    """Generation 0 means nothing held; the revoked marks also start at 0."""
    fresh = ControlAuthority(None, 0, 0, 0, 0, (), ())

    outcome = compare_control_state(fresh, peer(key_generation=0, spki_generation=0, recovery_generation=0))

    assert outcome is ControlComparison.NOT_PAIRED


def test_with_no_authority_a_peer_claiming_generations_the_add_on_never_issued_means_it_was_rolled_back() -> None:
    unpaired = local_state(authority=None, key_high_water=3, key_revoked_through=3, recovery_high_water=1)

    assert compare_control_state(unpaired, peer(key_generation=5)) is ControlComparison.LOCAL_BEHIND
    assert compare_control_state(unpaired, peer(key_generation=0, spki_generation=0, recovery_generation=2)) is (
        ControlComparison.LOCAL_BEHIND
    )
    assert compare_control_state(unpaired, peer(key_generation=0, spki_generation=0, recovery_generation=1)) is (
        ControlComparison.NOT_PAIRED
    )


@pytest.mark.parametrize("bad", [-1, 2**64, True, 1.5])
def test_malformed_peer_generations_are_value_errors(bad: Any) -> None:
    for field in ("key_generation", "spki_generation", "recovery_generation"):
        with pytest.raises(ValueError, match="unsigned 64-bit"):
            compare_control_state(local_state(), peer(**{field: bad}))


# --------------------------------------------------------------------------------------------------
# the quarantine record
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def keys(tmp_path: Path) -> StateAuthenticationKeyStore:
    store = StateAuthenticationKeyStore(tmp_path / "keys")
    store.create()
    return store


@pytest.fixture
def store(tmp_path: Path, keys: StateAuthenticationKeyStore) -> QuarantineStore:
    return QuarantineStore(tmp_path / "quarantine", keys=keys, clock=lambda: T0)


@pytest.fixture
def ready(store: QuarantineStore) -> QuarantineStore:
    """A store the add-on has initialized: the clear record exists."""
    store.create()
    return store


def refusal_of(operation: Callable[[], object]) -> QuarantineRefusal:
    with pytest.raises(QuarantineRefused) as caught:
        operation()
    return caught.value.reason


def fault_of(operation: Callable[[], object]) -> StoreFault:
    with pytest.raises(SecureStoreError) as caught:
        operation()
    return caught.value.fault


def declare_and_finish(store: QuarantineStore, reporters: list[str]) -> Quarantine:
    store.declare_restore(reporters)
    store.mark_control_repaired()
    store.mark_mqtt_reprovisioned()
    for item in reporters:
        store.dispose_reporter(item)
    return store.load()


def test_a_record_that_was_never_created_is_missing_and_missing_fails_closed(store: QuarantineStore) -> None:
    assert fault_of(store.load) is StoreFault.IDENTITY_MISSING
    decision = evaluate_gate(store)
    assert (decision.permitted, decision.reason) == (False, "quarantine_record_identity_missing")
    for operation in (
        store.mark_control_repaired,
        store.mark_mqtt_reprovisioned,
        lambda: store.dispose_reporter(reporter(1)),
        store.release,
    ):
        assert fault_of(operation) is StoreFault.IDENTITY_MISSING
    assert not store.path.exists()


def test_creating_the_record_makes_every_path_open(store: QuarantineStore) -> None:
    created = store.create()
    state = store.load()

    assert created == state
    assert not state.quarantined and state.restore_generation == 0
    assert state.normal_service_permitted and state.mqtt_v2_permitted and state.control_administration_permitted
    assert state.reporter_traffic_permitted
    assert state.outstanding == ()
    assert evaluate_gate(store).permitted
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_an_existing_record_is_never_replaced_by_create(ready: QuarantineStore) -> None:
    ready.declare_restore([reporter(1)])
    before = ready.path.read_bytes()

    assert fault_of(ready.create) is StoreFault.IDENTITY_EXISTS
    assert ready.path.read_bytes() == before


def test_a_record_created_by_another_writer_meanwhile_is_reported_as_existing(
    store: QuarantineStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keys: StateAuthenticationKeyStore
) -> None:
    QuarantineStore(tmp_path / "quarantine", keys=keys, clock=lambda: T0).create()
    winner = store.path.read_bytes()
    monkeypatch.setattr(AuthenticatedDocumentStore, "load", lambda self: None)

    assert fault_of(store.create) is StoreFault.IDENTITY_EXISTS
    assert store.path.read_bytes() == winner


def test_a_restore_that_left_no_record_can_still_be_declared(store: QuarantineStore) -> None:
    """The recovery path: the restored data had no marker, so declaring starts from nothing."""
    state = store.declare_restore([reporter(1)])

    assert state.quarantined and state.restore_generation == 1


def test_declaring_a_restore_quarantines_everything_restored(store: QuarantineStore) -> None:
    state = store.declare_restore([reporter(2), reporter(1)])

    assert state.quarantined and state.restore_generation == 1
    assert state.declared_at == int(T0.timestamp())
    assert state.pending_reporters == (reporter(1), reporter(2))  # sorted
    assert not state.normal_service_permitted
    assert not state.control_administration_permitted
    assert not state.mqtt_v2_permitted
    assert not state.reporter_traffic_permitted  # for every identity: even one missing from the declared list
    assert state.outstanding == (
        "control_repair",
        "mqtt_reprovision",
        f"reporter:{reporter(1)}",
        f"reporter:{reporter(2)}",
    )
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_the_state_survives_a_restart(
    store: QuarantineStore, tmp_path: Path, keys: StateAuthenticationKeyStore
) -> None:
    declared = store.declare_restore([reporter(1)])

    assert QuarantineStore(tmp_path / "quarantine", keys=keys, clock=lambda: T0).load() == declared


def test_duplicate_restored_reporters_collapse(store: QuarantineStore) -> None:
    assert store.declare_restore([reporter(1), reporter(1)]).pending_reporters == (reporter(1),)


def test_an_identity_missing_from_the_declared_list_is_blocked_too(store: QuarantineStore) -> None:
    """Reporter traffic is closed for everyone, not only for the identities that were listed."""
    state = store.declare_restore([reporter(1)])

    assert not state.reporter_traffic_permitted


def test_a_fresh_control_pairing_reopens_administration_but_nothing_else(store: QuarantineStore) -> None:
    store.declare_restore([reporter(1)])

    state = store.mark_control_repaired()

    assert state.control_administration_permitted  # the fresh authority is what deletes the restored identities
    assert not state.mqtt_v2_permitted and not state.normal_service_permitted
    assert not state.reporter_traffic_permitted
    assert state.outstanding == ("mqtt_reprovision", f"reporter:{reporter(1)}")


def test_deleting_a_restored_reporter_removes_it_from_the_pending_set_but_reopens_nothing(
    store: QuarantineStore,
) -> None:
    store.declare_restore([reporter(1), reporter(2)])
    store.mark_control_repaired()

    state = store.dispose_reporter(reporter(1))

    assert state.pending_reporters == (reporter(2),)
    assert not state.reporter_traffic_permitted  # traffic stays closed until the whole quarantine is released


@pytest.mark.parametrize("step", ["mqtt", "dispose"])
def test_mqtt_reprovisioning_and_deletion_need_the_fresh_pairing_first(store: QuarantineStore, step: str) -> None:
    store.declare_restore([reporter(1)])
    before = store.path.read_bytes()

    operation = store.mark_mqtt_reprovisioned if step == "mqtt" else lambda: store.dispose_reporter(reporter(1))

    assert refusal_of(operation) is QuarantineRefusal.CONTROL_NOT_REPAIRED
    assert store.path.read_bytes() == before


def test_disposing_a_reporter_is_idempotent_so_a_retry_after_a_crash_is_harmless(store: QuarantineStore) -> None:
    store.declare_restore([reporter(1)])
    store.mark_control_repaired()
    first = store.dispose_reporter(reporter(1))

    assert store.dispose_reporter(reporter(1)) == first
    assert store.dispose_reporter(reporter(7)) == first  # an identifier that was never pending changes nothing


def test_release_requires_every_obligation(store: QuarantineStore) -> None:
    store.declare_restore([reporter(1)])
    before = store.path.read_bytes()

    with pytest.raises(QuarantineRefused) as caught:
        store.release()
    assert caught.value.reason is QuarantineRefusal.OBLIGATIONS_OUTSTANDING
    assert "control_repair" in caught.value.detail and "mqtt_reprovision" in caught.value.detail
    assert f"reporter:{reporter(1)}" in caught.value.detail
    assert store.path.read_bytes() == before

    store.mark_control_repaired()
    assert refusal_of(store.release) is QuarantineRefusal.OBLIGATIONS_OUTSTANDING
    store.mark_mqtt_reprovisioned()
    assert refusal_of(store.release) is QuarantineRefusal.OBLIGATIONS_OUTSTANDING
    store.dispose_reporter(reporter(1))

    released = store.release()

    assert not released.quarantined and released.normal_service_permitted
    assert released.restore_generation == 1 and released.outstanding == ()
    assert released.pending_reporters == () and not released.control_repaired and not released.mqtt_reprovisioned


@pytest.mark.parametrize("missing", ["control", "mqtt", "reporter"])
def test_release_is_refused_while_any_single_obligation_is_open(store: QuarantineStore, missing: str) -> None:
    store.declare_restore([reporter(1)])
    if missing != "control":  # the later steps need the fresh pairing first
        store.mark_control_repaired()
        if missing != "mqtt":
            store.mark_mqtt_reprovisioned()
        if missing != "reporter":
            store.dispose_reporter(reporter(1))

    assert refusal_of(store.release) is QuarantineRefusal.OBLIGATIONS_OUTSTANDING
    assert store.load().quarantined


def test_a_restore_with_no_reporters_needs_only_pairing_and_mqtt(store: QuarantineStore) -> None:
    store.declare_restore([])
    store.mark_control_repaired()
    store.mark_mqtt_reprovisioned()

    assert not store.release().quarantined


def test_nothing_but_a_declaration_changes_a_clear_store(ready: QuarantineStore) -> None:
    before = ready.path.read_bytes()
    for operation in (
        ready.mark_control_repaired,
        ready.mark_mqtt_reprovisioned,
        lambda: ready.dispose_reporter(reporter(1)),
        ready.release,
    ):
        assert refusal_of(operation) is QuarantineRefusal.NOT_QUARANTINED
    assert ready.path.read_bytes() == before


def test_declaring_again_starts_over_so_progress_carried_in_restored_data_never_counts(store: QuarantineStore) -> None:
    declare_and_finish(store, [reporter(1)])
    store.release()
    store.declare_restore([reporter(1)])
    store.mark_control_repaired()
    store.mark_mqtt_reprovisioned()

    again = store.declare_restore([reporter(1), reporter(2)])  # for example a restore of a backup taken mid-quarantine

    assert again.restore_generation == 3
    assert not again.control_repaired and not again.mqtt_reprovisioned
    assert again.pending_reporters == (reporter(1), reporter(2))


def test_a_backup_taken_mid_quarantine_cannot_smuggle_progress_through_a_redeclared_restore(
    store: QuarantineStore, tmp_path: Path, keys: StateAuthenticationKeyStore
) -> None:
    store.declare_restore([reporter(1)])
    store.mark_control_repaired()
    store.mark_mqtt_reprovisioned()
    store.dispose_reporter(reporter(1))
    nearly_done = store.path.read_bytes()
    store.release()
    # Restoring the backup puts the near-complete record back; the declaration must still reset it.
    store.path.write_bytes(nearly_done)
    store.path.chmod(0o600)
    restored = QuarantineStore(tmp_path / "quarantine", keys=keys, clock=lambda: T0)

    state = restored.declare_restore([reporter(1)])

    assert state.outstanding == ("control_repair", "mqtt_reprovision", f"reporter:{reporter(1)}")


@pytest.mark.parametrize(
    "reporters",
    [["short"], [""], ["a" * 43 + "="], [reporter(1) + "x"], ["not-base64url!" * 4]],
)
def test_malformed_reporter_identifiers_are_value_errors_and_write_nothing(
    store: QuarantineStore, reporters: Any
) -> None:
    with pytest.raises(ValueError, match="canonical 43-character"):
        store.declare_restore(reporters)
    with pytest.raises(ValueError, match="canonical 43-character"):
        store.dispose_reporter(reporters[0])
    assert not store.path.exists()


def test_a_string_in_place_of_a_list_is_rejected(store: QuarantineStore) -> None:
    with pytest.raises(ValueError, match="canonical 43-character"):
        store.declare_restore(reporter(1))


def test_the_pending_set_is_capped_at_the_roster_size(store: QuarantineStore) -> None:
    ids = [encode_base64url(index.to_bytes(32, "big"), exact_length=32) for index in range(MAX_RESTORED_REPORTERS + 1)]

    assert len(store.declare_restore(ids[:MAX_RESTORED_REPORTERS]).pending_reporters) == MAX_RESTORED_REPORTERS
    with pytest.raises(ValueError, match="at most 1000"):
        store.declare_restore(ids)


def test_a_naive_clock_is_rejected(tmp_path: Path, keys: StateAuthenticationKeyStore) -> None:
    naive = QuarantineStore(tmp_path / "quarantine", keys=keys, clock=lambda: datetime(2030, 1, 1))  # noqa: DTZ001

    with pytest.raises(ValueError, match="timezone-aware"):
        naive.declare_restore([])


# --------------------------------------------------------------------------------------------------
# the gate fails closed
# --------------------------------------------------------------------------------------------------


def test_the_gate_permits_a_clear_store_and_blocks_a_quarantined_one(ready: QuarantineStore) -> None:
    store = ready
    clear = evaluate_gate(store)
    assert (clear.permitted, clear.reason) == (True, None)

    store.declare_restore([reporter(1)])
    blocked = evaluate_gate(store)

    assert (blocked.permitted, blocked.reason) == (False, "post_restore_quarantine")
    assert blocked.quarantine is not None and blocked.quarantine.quarantined


def test_the_gate_opens_again_only_after_release(store: QuarantineStore) -> None:
    declare_and_finish(store, [reporter(1)])
    assert not evaluate_gate(store).permitted

    store.release()

    assert evaluate_gate(store).permitted


def test_a_tampered_record_blocks_service_instead_of_reading_as_clear(
    store: QuarantineStore, tmp_path: Path, keys: StateAuthenticationKeyStore
) -> None:
    store.declare_restore([reporter(1)])
    store.path.write_bytes(store.path.read_bytes().replace(b'"state":"quarantined"', b'"state":"clear"      '))
    fresh = QuarantineStore(tmp_path / "quarantine", keys=keys, clock=lambda: T0)

    decision = evaluate_gate(fresh)

    assert decision.permitted is False
    assert decision.reason is not None and decision.reason.startswith("quarantine_record_")
    assert decision.quarantine is None


@pytest.mark.parametrize("mode", [0o644, 0o666])
def test_a_widened_record_blocks_service(store: QuarantineStore, mode: int) -> None:
    store.declare_restore([])
    store.path.chmod(mode)

    assert evaluate_gate(store).reason == "quarantine_record_file_insecure"


def test_a_record_without_its_state_key_blocks_service(tmp_path: Path, keys: StateAuthenticationKeyStore) -> None:
    QuarantineStore(tmp_path / "quarantine", keys=keys, clock=lambda: T0).declare_restore([])
    orphan = QuarantineStore(tmp_path / "quarantine", keys=StateAuthenticationKeyStore(tmp_path / "none"))

    assert evaluate_gate(orphan).reason == "quarantine_record_key_missing"


def second_writer(store: QuarantineStore, keys: StateAuthenticationKeyStore) -> AuthenticatedDocumentStore:
    return AuthenticatedDocumentStore(
        store.path.parent,
        DOCUMENT_NAME,
        store_id="post-restore-quarantine",
        schema_version=1,
        max_bytes=128 * 1024,
        keys=keys,
    )


def forge(store: QuarantineStore, keys: StateAuthenticationKeyStore, tmp_path: Path, **changes: Any) -> QuarantineStore:
    document = second_writer(store, keys)
    loaded = document.load()
    assert loaded is not None
    document.save({**loaded.record, **changes})
    return QuarantineStore(tmp_path / "quarantine", keys=keys, clock=lambda: T0)


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "bogus"},
        {"state": 1},
        {"control_repaired": "yes"},
        {"mqtt_reprovisioned": 1},
        {"declared_at": "now"},
        {"restore_generation": True},
        {"pending_reporters": "x"},
        {"pending_reporters": [1]},
        {"pending_reporters": ["short"]},
        {"pending_reporters": [reporter(2), reporter(1)]},  # not canonical order
        {"pending_reporters": [reporter(1), reporter(1)]},
        {"extra": 1},
    ],
)
def test_an_authentic_record_of_the_wrong_shape_blocks_service(
    store: QuarantineStore, keys: StateAuthenticationKeyStore, tmp_path: Path, changes: dict[str, Any]
) -> None:
    store.declare_restore([reporter(1)])
    fresh = forge(store, keys, tmp_path, **changes)

    assert fault_of(fresh.load) is StoreFault.MALFORMED
    assert evaluate_gate(fresh).permitted is False


def test_a_clear_record_that_carries_obligations_is_inconsistent(
    store: QuarantineStore, keys: StateAuthenticationKeyStore, tmp_path: Path
) -> None:
    store.declare_restore([reporter(1)])
    fresh = forge(store, keys, tmp_path, state="clear")

    assert fault_of(fresh.load) is StoreFault.IDENTITY_INCONSISTENT
    assert evaluate_gate(fresh).permitted is False


def test_a_quarantine_without_a_restore_generation_is_inconsistent(
    store: QuarantineStore, keys: StateAuthenticationKeyStore, tmp_path: Path
) -> None:
    store.declare_restore([reporter(1)])
    fresh = forge(store, keys, tmp_path, restore_generation=0)

    assert fault_of(fresh.load) is StoreFault.IDENTITY_INCONSISTENT


def test_a_refused_change_writes_nothing(store: QuarantineStore) -> None:
    store.declare_restore([reporter(1)])
    before = store.path.read_bytes()

    refusal_of(store.release)

    assert store.path.read_bytes() == before


def test_the_module_exports_its_public_names() -> None:
    for name in restore_guard.__all__:
        assert hasattr(restore_guard, name)
