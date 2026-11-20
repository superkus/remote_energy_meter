"""The reporter roster: lifecycle, stable IDs, key rotation, revocation, tombstones, capacities, the cursor."""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from remote_energy_meter.authenticated_store import StateAuthenticationKeyStore
from remote_energy_meter.protocol_v1 import encode_base64url, parse_p256_public_jwk
from remote_energy_meter.reporter_metadata import ReporterMetadata
from remote_energy_meter.roster import (
    MAX_REPORTERS,
    MAX_TOMBSTONES,
    TRANSIENT_EVENTS,
    ReporterRecord,
    ReporterState,
    Roster,
    RosterEvent,
    RosterRefusal,
    RosterRefused,
    RosterStore,
    Tombstone,
)
from remote_energy_meter.secure_store import SecureStoreError, StoreFault

T0 = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)
META = ReporterMetadata("Server A", "macos", "arm64", "1.0.0")
META_B = ReporterMetadata("Server B", "macos", "x86_64", "1.1.0")


def new_jwk() -> bytes:
    numbers = ec.generate_private_key(ec.SECP256R1()).public_key().public_numbers()
    member = {
        "crv": "P-256",
        "kty": "EC",
        "x": encode_base64url(numbers.x.to_bytes(32, "big"), exact_length=32),
        "y": encode_base64url(numbers.y.to_bytes(32, "big"), exact_length=32),
    }
    return json.dumps(member, separators=(",", ":"), sort_keys=True).encode("ascii")


def thumbprint(jwk: bytes) -> str:
    return parse_p256_public_jwk(jwk).thumbprint


def fake_id(n: int) -> str:
    return encode_base64url(n.to_bytes(32, "big"), exact_length=32)


@pytest.fixture
def keys(tmp_path: Path) -> StateAuthenticationKeyStore:
    store = StateAuthenticationKeyStore(tmp_path / "keys")
    store.create()
    return store


@pytest.fixture
def store(tmp_path: Path, keys: StateAuthenticationKeyStore) -> RosterStore:
    roster = RosterStore(tmp_path / "roster", keys=keys, clock=lambda: T0)
    roster.create()
    return roster


def enroll(store: RosterStore, metadata: ReporterMetadata = META) -> tuple[str, bytes]:
    jwk = new_jwk()
    store.activate(jwk, metadata)
    return thumbprint(jwk), jwk


def refusal(call: Callable[[], object]) -> RosterRefusal:
    with pytest.raises(RosterRefused) as caught:
        call()
    return caught.value.reason


def unchanged(store: RosterStore, call: Callable[[], object]) -> RosterRefusal:
    before = store.path.read_bytes()
    reason = refusal(call)
    assert store.path.read_bytes() == before
    return reason


def raw_record(store: RosterStore) -> dict[str, Any]:
    loaded = store._document.load()
    assert loaded is not None
    return copy.deepcopy(dict(loaded.record))


def forge(store: RosterStore, record: dict[str, Any]) -> None:
    """Save ``record`` correctly authenticated, as an attacker with the state key (or a bug) could."""
    store._document.save(record)


def fill(store: RosterStore, *, reporters: int = 0, tombstones: int = 0, generation: int = 5000) -> None:
    records = []
    for _ in range(reporters):
        jwk = new_jwk()
        records.append(
            ReporterRecord(
                reporter_id=thumbprint(jwk),
                state=ReporterState.ACTIVE,
                jwk=jwk,
                key_thumbprint=thumbprint(jwk),
                key_generation=1,
                state_generation=1,
                metadata=META,
                enrolled_at=1,
                revoked_at=None,
            )
        )
    stones = sorted((Tombstone(fake_id(n + 1), 2, 1) for n in range(tombstones)), key=lambda t: t.reporter_id)
    forge(
        store,
        dict(
            store._encode(
                Roster(
                    generation,
                    tuple(sorted(records, key=lambda r: r.reporter_id)),
                    tuple(stones),
                )
            )
        ),
    )


# --------------------------------------------------------------------------------------------------
# creation and loading
# --------------------------------------------------------------------------------------------------


def test_a_new_roster_is_empty_at_generation_zero(store: RosterStore) -> None:
    roster = store.load()

    assert (roster.generation, roster.reporters, roster.tombstones) == (0, (), ())


def test_a_missing_roster_fails_closed(tmp_path: Path, keys: StateAuthenticationKeyStore) -> None:
    store = RosterStore(tmp_path / "other", keys=keys)

    with pytest.raises(SecureStoreError) as caught:
        store.load()
    assert caught.value.fault is StoreFault.IDENTITY_MISSING
    with pytest.raises(SecureStoreError) as mutation:
        store.activate(new_jwk(), META)
    assert mutation.value.fault is StoreFault.IDENTITY_MISSING


def test_an_existing_roster_is_never_replaced(store: RosterStore) -> None:
    enroll(store)
    before = store.path.read_bytes()

    with pytest.raises(SecureStoreError) as caught:
        store.create()

    assert caught.value.fault is StoreFault.IDENTITY_EXISTS
    assert store.path.read_bytes() == before


def test_a_roster_created_by_another_writer_meanwhile_is_reported_as_existing(
    tmp_path: Path, keys: StateAuthenticationKeyStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RosterStore(tmp_path / "raced", keys=keys)

    def lose_the_race(_record: object) -> None:
        raise SecureStoreError(StoreFault.CONCURRENT_MODIFICATION, "another writer created it")

    monkeypatch.setattr(store._document, "create", lose_the_race)

    with pytest.raises(SecureStoreError) as caught:
        store.create()
    assert caught.value.fault is StoreFault.IDENTITY_EXISTS


# --------------------------------------------------------------------------------------------------
# activation
# --------------------------------------------------------------------------------------------------


def test_activation_uses_the_key_thumbprint_as_the_id_and_returns_the_event(store: RosterStore) -> None:
    jwk = new_jwk()

    change = store.activate(jwk, META)

    reporter_id = thumbprint(jwk)
    assert (change.generation, change.event, change.reporter_id) == (1, RosterEvent.REPORTER_ACTIVATED, reporter_id)
    assert change.reporter_state_generation == 1
    record = change.record
    assert record is not None
    assert (record.state, record.key_generation, record.state_generation) == (ReporterState.ACTIVE, 1, 1)
    assert (record.key_thumbprint, record.metadata, record.enrolled_at, record.revoked_at) == (
        reporter_id,
        META,
        int(T0.timestamp()),
        None,
    )
    assert store.load().get(reporter_id) == record
    assert store.load().generation == 1


def test_each_mutation_advances_the_cursor_by_exactly_one(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    store.update_metadata(reporter_id, META_B)
    store.revoke(reporter_id)
    store.delete(reporter_id)

    assert store.load().generation == 4


def test_activation_rejects_a_key_that_is_not_a_canonical_jwk(store: RosterStore) -> None:
    with pytest.raises(Exception, match=r"."):
        store.activate(b'{"kty":"EC"}', META)
    assert store.load().generation == 0


def test_an_enrolled_id_cannot_be_activated_again(store: RosterStore) -> None:
    _, jwk = enroll(store)

    assert unchanged(store, lambda: store.activate(jwk, META)) is RosterRefusal.EXISTS


def test_a_tombstoned_id_cannot_be_activated_until_the_tombstone_is_compacted(store: RosterStore) -> None:
    reporter_id, jwk = enroll(store)
    store.delete(reporter_id)

    assert unchanged(store, lambda: store.activate(jwk, META)) is RosterRefusal.TOMBSTONED

    generation = store.load().generation
    store.compact_tombstone(reporter_id)
    assert store.load().generation == generation  # internal bookkeeping, not a control-visible event
    again = store.activate(jwk, META)
    assert again.reporter_state_generation == 1


def test_a_key_that_is_another_reporters_rotated_key_cannot_be_activated(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    rotated = new_jwk()
    store.rotate_key(reporter_id, current_generation=1, proposed_jwk=rotated)

    assert unchanged(store, lambda: store.activate(rotated, META)) is RosterRefusal.KEY_IN_USE


def test_the_roster_holds_at_most_one_thousand_reporters(store: RosterStore) -> None:
    fill(store, reporters=MAX_REPORTERS)

    assert unchanged(store, lambda: store.activate(new_jwk(), META)) is RosterRefusal.CAPACITY_EXCEEDED
    assert len(store.load().reporters) == MAX_REPORTERS


def test_reporters_are_kept_sorted_by_id(store: RosterStore) -> None:
    for _ in range(5):
        enroll(store)

    ids = [r.reporter_id for r in store.load().reporters]
    assert ids == sorted(ids)


# --------------------------------------------------------------------------------------------------
# expected generation
# --------------------------------------------------------------------------------------------------


def test_a_stale_expected_generation_is_refused_without_writing(store: RosterStore) -> None:
    enroll(store)

    assert unchanged(store, lambda: store.activate(new_jwk(), META, expected_roster_generation=0)) is (
        RosterRefusal.STALE_GENERATION
    )


def test_the_current_expected_generation_is_accepted(store: RosterStore) -> None:
    enroll(store)

    change = store.activate(new_jwk(), META, expected_roster_generation=1)

    assert change.generation == 2


@pytest.mark.parametrize("bad", [-1, 2**64, True, "1"])
def test_a_malformed_expected_generation_is_a_caller_error(store: RosterStore, bad: Any) -> None:
    with pytest.raises(ValueError, match="expected_roster_generation"):
        store.activate(new_jwk(), META, expected_roster_generation=bad)


def test_every_mutation_honours_the_expected_generation(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    stale = 0
    calls: list[Callable[[], object]] = [
        lambda: store.update_metadata(reporter_id, META_B, expected_roster_generation=stale),
        lambda: store.rotate_key(
            reporter_id, current_generation=1, proposed_jwk=new_jwk(), expected_roster_generation=stale
        ),
        lambda: store.revoke(reporter_id, expected_roster_generation=stale),
        lambda: store.delete(reporter_id, expected_roster_generation=stale),
        lambda: store.record_transient_event(
            RosterEvent.PENDING_CREATED, reporter_id, 1, expected_roster_generation=stale
        ),
    ]

    for call in calls:
        assert unchanged(store, call) is RosterRefusal.STALE_GENERATION


def test_an_exhausted_cursor_refuses_further_changes(store: RosterStore) -> None:
    fill(store, generation=2**64 - 1)

    assert unchanged(store, lambda: store.activate(new_jwk(), META)) is RosterRefusal.GENERATION_EXHAUSTED


# --------------------------------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------------------------------


def test_changed_metadata_advances_the_reporter_and_roster_generations(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)

    change = store.update_metadata(reporter_id, META_B)

    assert change is not None
    assert (change.generation, change.event, change.reporter_state_generation) == (2, RosterEvent.METADATA_CHANGED, 2)
    assert change.record is not None
    assert change.record.metadata == META_B


def test_identical_metadata_changes_and_writes_nothing(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    before = store.path.read_bytes()

    assert store.update_metadata(reporter_id, META) is None

    assert store.path.read_bytes() == before


def test_metadata_of_an_unknown_or_revoked_reporter_is_refused(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)

    assert unchanged(store, lambda: store.update_metadata(fake_id(9), META_B)) is RosterRefusal.NOT_FOUND
    store.revoke(reporter_id)
    assert unchanged(store, lambda: store.update_metadata(reporter_id, META_B)) is RosterRefusal.NOT_ACTIVE


@pytest.mark.parametrize("bad", ["", "abc", "B" * 43, "=" * 43, "a" * 44])
def test_a_malformed_reporter_id_is_a_caller_error(store: RosterStore, bad: str) -> None:
    for call in (
        lambda: store.update_metadata(bad, META),
        lambda: store.revoke(bad),
        lambda: store.delete(bad),
        lambda: store.compact_tombstone(bad),
        lambda: store.rotate_key(bad, current_generation=1, proposed_jwk=new_jwk()),
        lambda: store.record_transient_event(RosterEvent.PENDING_CREATED, bad, 1),
    ):
        with pytest.raises(ValueError, match="reporter ID"):
            call()


# --------------------------------------------------------------------------------------------------
# key rotation
# --------------------------------------------------------------------------------------------------


def test_rotation_keeps_the_id_and_advances_the_key_generation(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    proposed = new_jwk()

    change = store.rotate_key(reporter_id, current_generation=1, proposed_jwk=proposed)

    record = change.record
    assert record is not None
    assert change.event is RosterEvent.KEY_ROTATED
    assert (record.reporter_id, record.key_generation, record.state_generation) == (reporter_id, 2, 2)
    assert record.key_thumbprint == thumbprint(proposed)
    assert record.jwk == parse_p256_public_jwk(proposed).canonical_bytes
    loaded = store.load()
    assert loaded.reporter_for_key(thumbprint(proposed)) == record
    assert loaded.get(reporter_id) == record


def test_the_old_key_stops_identifying_the_reporter_after_rotation(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    store.rotate_key(reporter_id, current_generation=1, proposed_jwk=new_jwk())

    assert store.load().reporter_for_key(reporter_id) is None


def test_a_second_rotation_needs_the_new_generation(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    store.rotate_key(reporter_id, current_generation=1, proposed_jwk=new_jwk())

    assert unchanged(store, lambda: store.rotate_key(reporter_id, current_generation=1, proposed_jwk=new_jwk())) is (
        RosterRefusal.KEY_GENERATION_MISMATCH
    )
    store.rotate_key(reporter_id, current_generation=2, proposed_jwk=new_jwk())
    assert store.load().get(reporter_id).key_generation == 3  # type: ignore[union-attr]


def test_a_rotated_reporter_remains_consistent_across_a_reload(store: RosterStore, tmp_path: Path) -> None:
    reporter_id, _ = enroll(store)
    store.rotate_key(reporter_id, current_generation=1, proposed_jwk=new_jwk())

    assert store.load().get(reporter_id) is not None


def test_rotation_refuses_a_key_the_roster_already_uses(store: RosterStore) -> None:
    first, first_jwk = enroll(store)
    second, _ = enroll(store)
    third_jwk = new_jwk()
    store.rotate_key(second, current_generation=1, proposed_jwk=third_jwk)

    # Another reporter's current key, its own current key, and a key equal to its own ID after rotation.
    assert unchanged(store, lambda: store.rotate_key(first, current_generation=1, proposed_jwk=third_jwk)) is (
        RosterRefusal.KEY_IN_USE
    )
    assert unchanged(store, lambda: store.rotate_key(first, current_generation=1, proposed_jwk=first_jwk)) is (
        RosterRefusal.KEY_IN_USE
    )


def test_rotation_refuses_the_key_of_an_enrolled_id_even_after_that_reporter_rotated(store: RosterStore) -> None:
    first, _ = enroll(store)
    second, second_jwk = enroll(store)
    store.rotate_key(second, current_generation=1, proposed_jwk=new_jwk())

    assert unchanged(store, lambda: store.rotate_key(first, current_generation=1, proposed_jwk=second_jwk)) is (
        RosterRefusal.KEY_IN_USE
    )


def test_rotation_refuses_the_key_of_a_tombstoned_id(store: RosterStore) -> None:
    first, _ = enroll(store)
    second, second_jwk = enroll(store)
    store.delete(second)

    assert unchanged(store, lambda: store.rotate_key(first, current_generation=1, proposed_jwk=second_jwk)) is (
        RosterRefusal.KEY_IN_USE
    )


def test_rotation_of_unknown_or_revoked_reporters_is_refused(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)

    assert unchanged(store, lambda: store.rotate_key(fake_id(3), current_generation=1, proposed_jwk=new_jwk())) is (
        RosterRefusal.NOT_FOUND
    )
    store.revoke(reporter_id)
    assert unchanged(store, lambda: store.rotate_key(reporter_id, current_generation=1, proposed_jwk=new_jwk())) is (
        RosterRefusal.NOT_ACTIVE
    )


@pytest.mark.parametrize("bad", [-1, 2**64, True, "1", 1.0])
def test_a_malformed_current_generation_is_a_caller_error(store: RosterStore, bad: Any) -> None:
    reporter_id, _ = enroll(store)

    with pytest.raises(ValueError, match="current_generation"):
        store.rotate_key(reporter_id, current_generation=bad, proposed_jwk=new_jwk())


# --------------------------------------------------------------------------------------------------
# revocation and deletion
# --------------------------------------------------------------------------------------------------


def test_revocation_keeps_the_record_and_stamps_the_time(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)

    change = store.revoke(reporter_id)

    record = change.record
    assert record is not None
    assert change.event is RosterEvent.REPORTER_REVOKED
    assert (record.state, record.revoked_at, record.state_generation) == (ReporterState.REVOKED, int(T0.timestamp()), 2)
    assert store.load().get(reporter_id) == record


def test_a_revoked_reporter_cannot_be_revoked_again_and_an_unknown_one_is_not_found(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    store.revoke(reporter_id)

    assert unchanged(store, lambda: store.revoke(reporter_id)) is RosterRefusal.NOT_ACTIVE
    assert unchanged(store, lambda: store.revoke(fake_id(7))) is RosterRefusal.NOT_FOUND


def test_a_revoked_reporters_key_remains_reserved(store: RosterStore) -> None:
    reporter_id, jwk = enroll(store)
    store.revoke(reporter_id)

    assert unchanged(store, lambda: store.activate(jwk, META)) is RosterRefusal.EXISTS


def test_deletion_leaves_a_tombstone_with_the_next_state_generation(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    store.update_metadata(reporter_id, META_B)

    change = store.delete(reporter_id)

    assert change.event is RosterEvent.REPORTER_DELETED
    assert change.record is None
    assert change.tombstone == Tombstone(reporter_id, 3, 3)
    assert change.reporter_state_generation == 3
    loaded = store.load()
    assert loaded.get(reporter_id) is None
    assert loaded.tombstone(reporter_id) == change.tombstone


def test_a_revoked_reporter_can_be_deleted(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    store.revoke(reporter_id)

    assert store.delete(reporter_id).tombstone is not None


def test_deleting_an_unknown_reporter_is_not_found(store: RosterStore) -> None:
    assert unchanged(store, lambda: store.delete(fake_id(1))) is RosterRefusal.NOT_FOUND


def test_a_delete_past_one_thousand_pending_cleanups_changes_nothing(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    record = raw_record(store)
    record["tombstones"] = sorted(
        (
            {"deleted_at_generation": 1, "reporter_id": fake_id(n + 1), "state_generation": 2}
            for n in range(MAX_TOMBSTONES)
        ),
        key=lambda t: t["reporter_id"],
    )
    forge(store, record)

    assert unchanged(store, lambda: store.delete(reporter_id)) is RosterRefusal.CAPACITY_EXCEEDED
    assert store.load().get(reporter_id) is not None


def test_compacting_an_unknown_tombstone_is_not_found(store: RosterStore) -> None:
    assert unchanged(store, lambda: store.compact_tombstone(fake_id(1))) is RosterRefusal.NOT_FOUND


def test_compaction_returns_the_new_roster_and_frees_capacity(store: RosterStore) -> None:
    reporter_id, _ = enroll(store)
    store.delete(reporter_id)

    roster = store.compact_tombstone(reporter_id)

    assert roster.tombstones == ()
    assert store.load() == roster


# --------------------------------------------------------------------------------------------------
# memory-only events
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("event", sorted(TRANSIENT_EVENTS, key=str))
def test_each_memory_only_event_advances_the_cursor_and_changes_no_record(
    store: RosterStore, event: RosterEvent
) -> None:
    reporter_id, _ = enroll(store)
    before = store.load()

    change = store.record_transient_event(event, reporter_id, 1)

    after = store.load()
    assert (change.generation, change.event, change.reporter_id, change.reporter_state_generation) == (
        2,
        event,
        reporter_id,
        1,
    )
    assert change.record is None and change.tombstone is None
    assert (after.reporters, after.tombstones) == (before.reporters, before.tombstones)
    assert after.generation == 2


@pytest.mark.parametrize("event", [e for e in RosterEvent if e not in TRANSIENT_EVENTS])
def test_a_durable_event_cannot_be_recorded_as_memory_only(store: RosterStore, event: RosterEvent) -> None:
    with pytest.raises(ValueError, match="memory-only"):
        store.record_transient_event(event, fake_id(1), 1)


def test_a_memory_only_event_needs_a_valid_state_generation(store: RosterStore) -> None:
    with pytest.raises(ValueError, match="reporter_state_generation"):
        store.record_transient_event(RosterEvent.PENDING_CREATED, fake_id(1), -1)


# --------------------------------------------------------------------------------------------------
# tampering and forged inconsistent records
# --------------------------------------------------------------------------------------------------


def duplicate_second(record: dict[str, Any]) -> None:
    record["reporters"][1] = copy.deepcopy(record["reporters"][0])


def unsort(record: dict[str, Any]) -> None:
    record["reporters"].reverse()


def share_key(record: dict[str, Any]) -> None:
    first = record["reporters"][0]
    record["reporters"][1]["jwk"] = first["jwk"]
    record["reporters"][1]["key_thumbprint"] = first["key_thumbprint"]


def revoked_without_time(record: dict[str, Any]) -> None:
    record["reporters"][0]["state"] = "revoked"


def time_without_revocation(record: dict[str, Any]) -> None:
    record["reporters"][0]["revoked_at"] = 5


def ahead_of_roster(record: dict[str, Any]) -> None:
    record["reporters"][0]["state_generation"] = record["generation"] + 1


def below_key_generation(record: dict[str, Any]) -> None:
    record["reporters"][0]["key_generation"] = 3


def unrotated_under_other_key(record: dict[str, Any]) -> None:
    other = record["reporters"][1]
    record["reporters"][0]["jwk"] = other["jwk"]
    record["reporters"][0]["key_thumbprint"] = other["key_thumbprint"]
    record["reporters"][1]["key_generation"] = 2
    record["reporters"][1]["state_generation"] = 2
    record["reporters"][1]["jwk"] = new_jwk().decode("ascii")
    record["reporters"][1]["key_thumbprint"] = thumbprint(record["reporters"][1]["jwk"].encode("ascii"))


def enrolled_and_tombstoned(record: dict[str, Any]) -> None:
    record["tombstones"] = [
        {"deleted_at_generation": 1, "reporter_id": record["reporters"][0]["reporter_id"], "state_generation": 2}
    ]


def duplicate_tombstone(record: dict[str, Any]) -> None:
    stone = {"deleted_at_generation": 1, "reporter_id": fake_id(1), "state_generation": 2}
    record["tombstones"] = [stone, dict(stone)]


def tombstone_ahead(record: dict[str, Any]) -> None:
    record["tombstones"] = [
        {"deleted_at_generation": record["generation"] + 1, "reporter_id": fake_id(1), "state_generation": 2}
    ]


def tombstone_without_state_generation(record: dict[str, Any]) -> None:
    record["tombstones"] = [{"deleted_at_generation": 1, "reporter_id": fake_id(1), "state_generation": 0}]


@pytest.mark.parametrize(
    "tamper",
    [
        duplicate_second,
        unsort,
        share_key,
        revoked_without_time,
        time_without_revocation,
        ahead_of_roster,
        below_key_generation,
        unrotated_under_other_key,
        enrolled_and_tombstoned,
        duplicate_tombstone,
        tombstone_ahead,
        tombstone_without_state_generation,
    ],
)
def test_an_authentic_record_that_breaks_a_roster_rule_is_refused(
    store: RosterStore, tamper: Callable[..., None]
) -> None:
    enroll(store)
    enroll(store)
    record = raw_record(store)
    tamper(record)
    forge(store, record)

    with pytest.raises(SecureStoreError) as caught:
        store.load()
    assert caught.value.fault in {StoreFault.IDENTITY_INCONSISTENT, StoreFault.MALFORMED}
    with pytest.raises(SecureStoreError):
        store.activate(new_jwk(), META)


def wrong_thumbprint(record: dict[str, Any]) -> None:
    record["reporters"][0]["key_thumbprint"] = fake_id(1)


def extra_reporter_field(record: dict[str, Any]) -> None:
    record["reporters"][0]["extra"] = 1


def missing_reporter_field(record: dict[str, Any]) -> None:
    del record["reporters"][0]["state"]


def unknown_state(record: dict[str, Any]) -> None:
    record["reporters"][0]["state"] = "pending"


def bad_metadata(record: dict[str, Any]) -> None:
    record["reporters"][0]["metadata"]["platform"] = "plan9"


def bad_jwk(record: dict[str, Any]) -> None:
    record["reporters"][0]["jwk"] = '{"kty":"EC"}'


def non_ascii_jwk(record: dict[str, Any]) -> None:
    record["reporters"][0]["jwk"] = "é"


def bad_id(record: dict[str, Any]) -> None:
    record["reporters"][0]["reporter_id"] = "short"


@pytest.mark.parametrize("field", ["key_generation", "state_generation", "enrolled_at"])
@pytest.mark.parametrize("value", [True, "1"])
def test_a_reporter_number_outside_uint64_is_refused(store: RosterStore, field: str, value: Any) -> None:
    enroll(store)
    record = raw_record(store)
    record["reporters"][0][field] = value
    forge(store, record)

    with pytest.raises(SecureStoreError):
        store.load()


def test_malformed_reporter_records_are_refused(store: RosterStore) -> None:
    enroll(store)
    good = raw_record(store)
    tampers: list[Callable[[dict[str, Any]], None]] = [
        wrong_thumbprint,
        extra_reporter_field,
        missing_reporter_field,
        unknown_state,
        bad_metadata,
        bad_jwk,
        non_ascii_jwk,
        bad_id,
        lambda r: r["reporters"].__setitem__(0, "x"),
        lambda r: r["reporters"][0].__setitem__("revoked_at", "now"),
        lambda r: r.__setitem__("generation", True),
        lambda r: r.__setitem__("reporters", {}),
        lambda r: r.__setitem__("tombstones", None),
        lambda r: r.__setitem__("extra", 1),
        lambda r: r.pop("tombstones"),
        lambda r: r.__setitem__("tombstones", ["x"]),
        lambda r: r.__setitem__("tombstones", [{"reporter_id": fake_id(1)}]),
        lambda r: r.__setitem__(
            "tombstones", [{"deleted_at_generation": "1", "reporter_id": fake_id(1), "state_generation": 2}]
        ),
        lambda r: r.__setitem__(
            "tombstones", [{"deleted_at_generation": 1, "reporter_id": "short", "state_generation": 2}]
        ),
        lambda r: r.__setitem__("reporters", [r["reporters"][0]] * (MAX_REPORTERS + 1)),
        lambda r: r.__setitem__(
            "tombstones", [r["tombstones"]] * (MAX_TOMBSTONES + 1) if r["tombstones"] else [{}] * (MAX_TOMBSTONES + 1)
        ),
    ]
    for tamper in tampers:
        record = copy.deepcopy(good)
        tamper(record)
        forge(store, record)
        with pytest.raises(SecureStoreError):
            store.load()


# --------------------------------------------------------------------------------------------------
# concurrency
# --------------------------------------------------------------------------------------------------


def test_concurrent_activations_each_get_a_distinct_contiguous_generation(store: RosterStore) -> None:
    writers = [store] * 6
    results: list[int] = []
    errors: list[BaseException] = []

    def run(writer: RosterStore) -> None:
        try:
            results.append(writer.activate(new_jwk(), META).generation)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(w,)) for w in writers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(results) == [1, 2, 3, 4, 5, 6]
    assert len(store.load().reporters) == 6


# --------------------------------------------------------------------------------------------------
# clock
# --------------------------------------------------------------------------------------------------


def test_a_clock_before_1970_is_refused_clearly(tmp_path: Path, keys: StateAuthenticationKeyStore) -> None:
    store = RosterStore(tmp_path / "old", keys=keys, clock=lambda: datetime(1969, 12, 31, tzinfo=UTC))
    store.create()

    with pytest.raises(ValueError, match="1970"):
        store.activate(new_jwk(), META)
    with pytest.raises(ValueError, match="1970"):
        store.revoke(fake_id(1))


def test_a_signing_key_equal_to_another_reporters_id_is_refused_on_load(store: RosterStore) -> None:
    first, _ = enroll(store)
    second, _ = enroll(store)
    record = raw_record(store)
    by_id = {r["reporter_id"]: r for r in record["reporters"]}
    by_id[second]["jwk"] = by_id[first]["jwk"]
    by_id[second]["key_thumbprint"] = first
    by_id[second]["key_generation"] = 2
    by_id[second]["state_generation"] = 2
    by_id[first]["jwk"] = new_jwk().decode("ascii")
    by_id[first]["key_thumbprint"] = thumbprint(by_id[first]["jwk"].encode("ascii"))
    by_id[first]["key_generation"] = 2
    by_id[first]["state_generation"] = 2
    forge(store, record)

    with pytest.raises(SecureStoreError) as caught:
        store.load()
    assert caught.value.fault is StoreFault.IDENTITY_INCONSISTENT


def test_a_naive_clock_is_refused(tmp_path: Path, keys: StateAuthenticationKeyStore) -> None:
    store = RosterStore(tmp_path / "naive", keys=keys, clock=lambda: datetime(2030, 1, 1))  # noqa: DTZ001
    store.create()

    with pytest.raises(ValueError, match="clock"):
        store.activate(new_jwk(), META)
