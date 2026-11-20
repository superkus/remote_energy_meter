"""Pure retention tables: replay nonces, exact wire retries, and 24-hour/30-day operation results."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import pytest

from remote_energy_meter import retention_tables
from remote_energy_meter.protocol_v1 import encode_base64url, render_canonical_json
from remote_energy_meter.retention_tables import (
    FULL_RESULT_SECONDS,
    MARKER_SECONDS,
    MAX_BODY_BYTES,
    MAX_FULL_BODY_BYTES,
    MAX_NONCES,
    MAX_OPERATIONS,
    MAX_REQUEST_LIFETIME_SECONDS,
    MAX_RETRIES,
    ExactRetry,
    ExactRetryTable,
    NonceTable,
    OperationResult,
    OperationResultTable,
    OperationState,
    TableRefusal,
    TableRefused,
)

NOW = 1_900_000_000


def nonce(n: int) -> bytes:
    return bytes([n % 256]) * 16 + b""


def ident(n: int) -> str:
    return encode_base64url(bytes([n]) * 16, exact_length=16)


def digest(n: int) -> bytes:
    return hashlib.sha256(bytes([n])).digest()


def refusal_of(operation: Any) -> TableRefusal:
    with pytest.raises(TableRefused) as caught:
        operation()
    return caught.value.reason


def retry(n: int = 1, **overrides: Any) -> ExactRetry:
    values: dict[str, Any] = {
        "purpose": "rem-control-key-rotation-v1",
        "request_hash": digest(n),
        "nonce": nonce(n),
        "key_generation": 3,
        "proposed_generation": 4,
        "status": 200,
        "body": b'{"ok":true}',
        "expires_at": NOW + 60,
    }
    values.update(overrides)
    return ExactRetry(**values)


def result(n: int = 1, *, committed_at: int = NOW, **overrides: Any) -> OperationResult:
    values: dict[str, Any] = {
        "operation_id": ident(n),
        "digest": digest(n),
        "status": 200,
        "body": b'{"state":"revoked"}',
        "committed_at": committed_at,
    }
    values.update(overrides)
    return OperationResult(**values)


# --------------------------------------------------------------------------------------------------
# nonces
# --------------------------------------------------------------------------------------------------


def test_a_nonce_is_accepted_once_and_a_replay_is_refused() -> None:
    table = NonceTable().consume(nonce(1), NOW + 60, NOW)

    assert table.contains(nonce(1), NOW)
    assert refusal_of(lambda: table.consume(nonce(1), NOW + 60, NOW + 1)) is TableRefusal.REPLAYED
    assert table.consume(nonce(2), NOW + 60, NOW).contains(nonce(2), NOW)


def test_consuming_never_changes_the_table_it_started_from() -> None:
    original = NonceTable()

    original.consume(nonce(1), NOW + 60, NOW)

    assert original == NonceTable()


@pytest.mark.parametrize(("expires_at", "accepted"), [(NOW - 1, False), (NOW, True), (NOW + 1, True)])
def test_a_request_is_valid_up_to_and_including_its_expiry(expires_at: int, accepted: bool) -> None:
    if accepted:
        assert NonceTable().consume(nonce(1), expires_at, NOW).contains(nonce(1), NOW)
    else:
        assert refusal_of(lambda: NonceTable().consume(nonce(1), expires_at, NOW)) is TableRefusal.EXPIRED


@pytest.mark.parametrize(("offset", "retained"), [(-1, True), (0, True), (1, False)])
def test_a_nonce_is_retained_through_its_expiry_instant_and_dropped_after(offset: int, retained: bool) -> None:
    table = NonceTable().consume(nonce(1), NOW + 60, NOW)
    at = NOW + 60 + offset

    assert table.contains(nonce(1), at) is retained
    assert (nonce(1) in {n for n, _ in table.compact(at).entries}) is retained


def test_a_replay_inside_the_window_is_refused_at_its_last_valid_instant() -> None:
    table = NonceTable().consume(nonce(1), NOW + 60, NOW)

    assert refusal_of(lambda: table.consume(nonce(1), NOW + 60, NOW + 60)) is TableRefusal.REPLAYED


def test_an_expired_nonce_is_forgotten_so_the_same_value_may_be_used_by_a_later_request() -> None:
    table = NonceTable().consume(nonce(1), NOW + 60, NOW)

    later = table.consume(nonce(1), NOW + 150, NOW + 61)

    assert later.entries == ((nonce(1), NOW + 150),)


def test_the_capacity_counts_only_unexpired_nonces_and_never_evicts() -> None:
    table = NonceTable()
    for index in range(MAX_NONCES):
        table = table.consume(index.to_bytes(16, "big"), NOW + 60, NOW)
    full = table

    assert refusal_of(lambda: full.consume(b"\xff" * 16, NOW + 60, NOW)) is TableRefusal.CAPACITY_EXCEEDED
    assert len(full.entries) == MAX_NONCES
    after = full.consume(b"\xff" * 16, NOW + 150, NOW + 61)  # every earlier nonce has expired and is compacted away
    assert after.entries == ((b"\xff" * 16, NOW + 150),)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((b"short", NOW + 60, NOW), "nonce"),
        ((nonce(1), -1, NOW), "expires_at"),
        ((nonce(1), NOW + 60, 2**64), "now"),
        ((nonce(1), True, NOW), "expires_at"),
    ],
)
def test_malformed_nonce_arguments_are_value_errors(args: tuple[Any, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NonceTable().consume(*args)


def test_nonce_tables_round_trip_through_json_and_decoding_is_strict() -> None:
    table = NonceTable().consume(nonce(1), NOW + 60, NOW).consume(nonce(2), NOW + 30, NOW)

    assert NonceTable.from_json(table.to_json()) == table
    good = table.to_json()
    first: dict[str, Any] = dict(good[0])  # type: ignore[arg-type]
    for bad in (
        "no",
        [1],
        [{**first, "extra": 1}],
        [{"nonce": first["nonce"]}],
        [{**first, "expires_at": "1"}],
        [{**first, "nonce": "short"}],
        [{**first, "expires_at": True}],
        [good[0], good[0]],
        [first] * (MAX_NONCES + 1),
    ):
        with pytest.raises(ValueError):
            NonceTable.from_json(bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------------
# exact wire retries
# --------------------------------------------------------------------------------------------------


def lookup(table: ExactRetryTable, entry: ExactRetry, now: int = NOW, **overrides: Any) -> ExactRetry | None:
    arguments: dict[str, Any] = {
        "purpose": entry.purpose,
        "request_hash": entry.request_hash,
        "nonce": entry.nonce,
        "key_generation": entry.key_generation,
        "proposed_generation": entry.proposed_generation,
        "now": now,
    }
    arguments.update(overrides)
    return table.lookup(**arguments)


def test_an_exact_retry_returns_the_recorded_result() -> None:
    entry = retry()
    table = ExactRetryTable().record(entry, NOW)

    found = lookup(table, entry)

    assert found is not None and (found.status, found.body) == (200, b'{"ok":true}')


@pytest.mark.parametrize(
    "override",
    [
        {"purpose": "rem-other-v1"},
        {"request_hash": digest(99)},
        {"nonce": nonce(99)},
        {"key_generation": 4},
        {"proposed_generation": 5},
        {"proposed_generation": None},
    ],
)
def test_every_bound_field_must_match_exactly(override: dict[str, Any]) -> None:
    entry = retry()
    table = ExactRetryTable().record(entry, NOW)

    assert lookup(table, entry, **override) is None


def test_a_record_without_a_proposed_generation_matches_only_none() -> None:
    entry = retry(proposed_generation=None)
    table = ExactRetryTable().record(entry, NOW)

    assert lookup(table, entry) is not None
    assert lookup(table, entry, proposed_generation=4) is None


@pytest.mark.parametrize(("offset", "found"), [(-1, True), (0, True), (1, False)])
def test_a_retry_is_returnable_through_its_expiry_instant_and_not_after(offset: int, found: bool) -> None:
    entry = retry()
    table = ExactRetryTable().record(entry, NOW)

    assert (lookup(table, entry, NOW + 60 + offset) is not None) is found


def test_a_second_result_for_the_same_request_is_refused_until_the_first_expires() -> None:
    table = ExactRetryTable().record(retry(), NOW)

    assert refusal_of(lambda: table.record(retry(nonce=nonce(7)), NOW + 1)) is TableRefusal.EXISTS
    assert table.record(retry(expires_at=NOW + 150), NOW + 61).entries[0].expires_at == NOW + 150


def test_a_result_for_an_already_expired_request_is_refused() -> None:
    assert refusal_of(lambda: ExactRetryTable().record(retry(expires_at=NOW - 1), NOW)) is TableRefusal.EXPIRED
    assert ExactRetryTable().record(retry(expires_at=NOW), NOW).entries


def test_the_retry_capacity_is_bounded() -> None:
    table = ExactRetryTable()
    for index in range(MAX_RETRIES):
        table = table.record(retry(request_hash=index.to_bytes(32, "big")), NOW)

    assert refusal_of(lambda: table.record(retry(request_hash=b"\xff" * 32), NOW)) is TableRefusal.CAPACITY_EXCEEDED


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"purpose": "Bad Purpose"}, "purpose"),
        ({"purpose": ""}, "purpose"),
        ({"purpose": "a" * 65}, "purpose"),
        ({"request_hash": b"short"}, "request_hash"),
        ({"nonce": b"short"}, "nonce"),
        ({"key_generation": -1}, "key_generation"),
        ({"proposed_generation": 2**64}, "proposed_generation"),
        ({"status": 99}, "status"),
        ({"status": 600}, "status"),
        ({"body": b""}, "body"),
        ({"body": b"x" * (MAX_BODY_BYTES + 1)}, "body"),
        ({"expires_at": True}, "expires_at"),
    ],
)
def test_malformed_retry_fields_are_value_errors(override: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExactRetryTable().record(retry(**override), NOW)


def test_a_recorded_body_never_appears_in_repr() -> None:
    entry = retry(body=b"SECRET-BODY-MARKER")

    assert "SECRET-BODY-MARKER" not in repr(entry)
    assert "SECRET-BODY-MARKER" not in repr(ExactRetryTable().record(entry, NOW))


def test_retry_tables_round_trip_through_json_and_decoding_is_strict() -> None:
    table = ExactRetryTable().record(retry(1), NOW).record(retry(2, proposed_generation=None), NOW)

    assert ExactRetryTable.from_json(table.to_json()) == table
    first: dict[str, Any] = dict(table.to_json()[0])  # type: ignore[arg-type]
    for bad in (
        "no",
        [{**first, "extra": 1}],
        [{**first, "status": "200"}],
        [{**first, "status": 99}],
        [{**first, "purpose": "BAD"}],
        [{**first, "body": ""}],
        [{**first, "proposed_generation": "4"}],
        [{k: v for k, v in first.items() if k != "nonce"}],
        [first, {**first, "nonce": encode_base64url(b"\x09" * 16, exact_length=16)}],
    ):
        with pytest.raises(ValueError):
            ExactRetryTable.from_json(bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------------
# operation results
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (-1, OperationState.FULL),
        (0, OperationState.FULL),
        (FULL_RESULT_SECONDS - 1, OperationState.FULL),
        (FULL_RESULT_SECONDS, OperationState.MARKER),
        (FULL_RESULT_SECONDS + MARKER_SECONDS - 1, OperationState.MARKER),
        (FULL_RESULT_SECONDS + MARKER_SECONDS, OperationState.GONE),
        (FULL_RESULT_SECONDS + MARKER_SECONDS + 1, OperationState.GONE),
    ],
)
def test_the_windows_are_half_open_at_exactly_twenty_four_hours_and_thirty_more_days(
    offset: int, expected: OperationState
) -> None:
    assert result().state(NOW + offset) is expected


def test_an_exact_retry_in_the_full_window_returns_the_original_status_and_body() -> None:
    table = OperationResultTable().record(result(), NOW)

    answer = table.lookup(ident(1), digest(1), NOW + FULL_RESULT_SECONDS - 1)

    assert (answer.state, answer.conflict, answer.status, answer.body) == (
        OperationState.FULL,
        False,
        200,
        b'{"state":"revoked"}',
    )


def test_an_exact_retry_in_the_marker_window_is_expired_without_a_body() -> None:
    table = OperationResultTable().record(result(), NOW)

    answer = table.lookup(ident(1), digest(1), NOW + FULL_RESULT_SECONDS)

    assert (answer.state, answer.conflict, answer.status, answer.body) == (OperationState.MARKER, False, None, None)


@pytest.mark.parametrize(
    "offset", [0, FULL_RESULT_SECONDS - 1, FULL_RESULT_SECONDS, FULL_RESULT_SECONDS + MARKER_SECONDS - 1]
)
def test_rebinding_an_identifier_to_another_digest_is_a_conflict_in_both_windows(offset: int) -> None:
    table = OperationResultTable().record(result(), NOW)

    answer = table.lookup(ident(1), digest(99), NOW + offset)

    assert answer.conflict is True and answer.body is None


@pytest.mark.parametrize("digest_matches", [True, False])
def test_after_the_marker_window_the_operation_is_ignored(digest_matches: bool) -> None:
    table = OperationResultTable().record(result(), NOW)

    answer = table.lookup(
        ident(1), digest(1) if digest_matches else digest(99), NOW + FULL_RESULT_SECONDS + MARKER_SECONDS
    )

    assert (answer.state, answer.conflict, answer.body) == (OperationState.GONE, False, None)


def test_an_unknown_identifier_is_ignored() -> None:
    answer = OperationResultTable().record(result(), NOW).lookup(ident(2), digest(2), NOW)

    assert (answer.state, answer.conflict) == (OperationState.GONE, False)


def test_a_lagging_cleanup_never_returns_a_body_after_its_boundary() -> None:
    """The full record is still physically present at 24 hours; the lookup still refuses to return it."""
    table = OperationResultTable().record(result(), NOW)
    assert table.entries[0].body is not None

    answer = table.lookup(ident(1), digest(1), NOW + FULL_RESULT_SECONDS)

    assert answer.body is None and answer.state is OperationState.MARKER


def test_a_lagging_cleanup_never_keeps_a_marker_past_its_end() -> None:
    marker = result(body=None)
    table = OperationResultTable((marker,))

    answer = table.lookup(ident(1), digest(99), NOW + FULL_RESULT_SECONDS + MARKER_SECONDS)

    assert (answer.state, answer.conflict) == (OperationState.GONE, False)


def test_compaction_deletes_the_body_at_twenty_four_hours_and_the_entry_after_the_marker_window() -> None:
    table = OperationResultTable().record(result(), NOW)

    assert table.compact(NOW + FULL_RESULT_SECONDS - 1).entries == table.entries
    marker = table.compact(NOW + FULL_RESULT_SECONDS).entries
    assert marker == (replace(result(), body=None),)
    assert marker[0].digest == digest(1)  # the marker keeps the digest, so a rebind is still a conflict
    assert table.compact(NOW + FULL_RESULT_SECONDS + MARKER_SECONDS - 1).entries == marker
    assert table.compact(NOW + FULL_RESULT_SECONDS + MARKER_SECONDS).entries == ()


def test_a_known_identifier_cannot_be_recorded_again_until_it_is_gone() -> None:
    table = OperationResultTable().record(result(), NOW)

    for offset in (1, FULL_RESULT_SECONDS, FULL_RESULT_SECONDS + MARKER_SECONDS - 1):
        later = NOW + offset
        assert refusal_of(lambda at=later: table.record(result(committed_at=at), at)) is TableRefusal.EXISTS
    gone = NOW + FULL_RESULT_SECONDS + MARKER_SECONDS
    renewed = table.record(result(digest=digest(50), committed_at=gone), gone)
    assert [e.digest for e in renewed.entries] == [digest(50)]


def test_the_operation_capacity_counts_only_retained_results() -> None:
    table = OperationResultTable()
    for index in range(MAX_OPERATIONS):
        table = table.record(result(operation_id=encode_base64url(index.to_bytes(16, "big"), exact_length=16)), NOW)

    extra = result(operation_id=ident(250))
    assert refusal_of(lambda: table.record(extra, NOW)) is TableRefusal.CAPACITY_EXCEEDED
    gone = NOW + FULL_RESULT_SECONDS + MARKER_SECONDS
    assert len(table.record(replace(extra, committed_at=gone), gone).entries) == 1


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"operation_id": "short"}, "operation_id"),
        ({"digest": b"short"}, "digest"),
        ({"status": 99}, "status"),
        ({"body": None}, "body"),
        ({"body": b""}, "body"),
        ({"body": b"x" * (MAX_BODY_BYTES + 1)}, "body"),
        ({"committed_at": -1}, "committed_at"),
        ({"committed_at": NOW + 1}, "recorded"),
    ],
)
def test_malformed_operation_results_are_value_errors(override: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OperationResultTable().record(result(**override), NOW)


def test_a_recorded_operation_body_never_appears_in_repr() -> None:
    entry = result(body=b"SECRET-BODY-MARKER")

    assert "SECRET-BODY-MARKER" not in repr(entry)


def test_operation_tables_round_trip_including_markers_and_decoding_is_strict() -> None:
    table = OperationResultTable((result(1), result(2, body=None)))

    assert OperationResultTable.from_json(table.to_json()) == table
    first: dict[str, Any] = dict(table.to_json()[0])  # type: ignore[arg-type]
    for bad in (
        "no",
        [{**first, "extra": 1}],
        [{**first, "status": "200"}],
        [{**first, "status": 99}],
        [{**first, "id": "short"}],
        [{**first, "body": ""}],
        [{**first, "committed_at": "1"}],
        [{**first, "digest": "short"}],
        [first, first],
    ):
        with pytest.raises(ValueError):
            OperationResultTable.from_json(bad)  # type: ignore[arg-type]


def test_the_module_exports_its_public_names_and_encodes_to_canonical_json() -> None:
    for name in retention_tables.__all__:
        assert hasattr(retention_tables, name)
    table = OperationResultTable().record(result(), NOW)
    assert render_canonical_json({"results": table.to_json()}, max_depth=6)


# --------------------------------------------------------------------------------------------------
# bounded expiry and bounded size
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("offset", "accepted"), [(0, True), (MAX_REQUEST_LIFETIME_SECONDS, True)])
def test_a_request_expiry_up_to_the_longest_valid_lifetime_is_accepted(offset: int, accepted: bool) -> None:
    assert NonceTable().consume(nonce(1), NOW + offset, NOW).entries
    assert ExactRetryTable().record(retry(expires_at=NOW + offset), NOW).entries
    assert accepted


@pytest.mark.parametrize("offset", [MAX_REQUEST_LIFETIME_SECONDS + 1, 10_000, 2**64 - 1 - NOW])
def test_a_request_expiry_beyond_the_longest_valid_lifetime_is_rejected(offset: int) -> None:
    with pytest.raises(ValueError, match="more than 90 seconds"):
        NonceTable().consume(nonce(1), NOW + offset, NOW)
    with pytest.raises(ValueError, match="more than 90 seconds"):
        ExactRetryTable().record(retry(expires_at=NOW + offset), NOW)


def test_a_hostile_far_future_expiry_cannot_pin_a_slot() -> None:
    table = NonceTable()
    for index in range(MAX_NONCES):
        table = table.consume(index.to_bytes(16, "big"), NOW + MAX_REQUEST_LIFETIME_SECONDS, NOW)

    # Every legitimate entry is gone ninety-one seconds later, so the table is usable again.
    later = NOW + MAX_REQUEST_LIFETIME_SECONDS + 1
    assert table.consume(b"\xee" * 16, later + 5, later).entries == ((b"\xee" * 16, later + 5),)


def full_result(n: int, size: int) -> OperationResult:
    return result(operation_id=encode_base64url(n.to_bytes(16, "big"), exact_length=16), body=b"x" * size)


def test_the_bytes_held_in_full_operation_results_are_capped_and_freed_at_the_marker_boundary() -> None:
    table = OperationResultTable()
    per = MAX_FULL_BODY_BYTES // MAX_BODY_BYTES
    for index in range(per):
        table = table.record(full_result(index, MAX_BODY_BYTES), NOW)

    assert refusal_of(lambda: table.record(full_result(per, 1), NOW)) is TableRefusal.CAPACITY_EXCEEDED
    later = NOW + FULL_RESULT_SECONDS
    freed = table.record(replace(full_result(per, 1), committed_at=later), later)
    assert len(freed.entries) == per + 1  # the earlier results became markers and no longer count


def test_a_decoded_table_may_not_hold_more_full_body_bytes_than_the_cap() -> None:
    per = MAX_FULL_BODY_BYTES // MAX_BODY_BYTES
    table = OperationResultTable(tuple(full_result(i, MAX_BODY_BYTES) for i in range(per + 1)))

    with pytest.raises(ValueError, match="more bytes"):
        OperationResultTable.from_json(table.to_json())


def test_all_three_tables_at_capacity_fit_comfortably_inside_one_document() -> None:
    nonces = NonceTable()
    for index in range(MAX_NONCES):
        nonces = nonces.consume(index.to_bytes(16, "big"), NOW + MAX_REQUEST_LIFETIME_SECONDS, NOW)
    retries = ExactRetryTable()
    for index in range(MAX_RETRIES):
        retries = retries.record(retry(request_hash=index.to_bytes(32, "big"), body=b"x" * MAX_BODY_BYTES), NOW)
    operations = OperationResultTable()
    batch = MAX_FULL_BODY_BYTES // MAX_BODY_BYTES
    clock = NOW
    for start in range(0, MAX_OPERATIONS, batch):
        for index in range(start, min(start + batch, MAX_OPERATIONS)):
            operations = operations.record(replace(full_result(index, MAX_BODY_BYTES), committed_at=clock), clock)
        clock += FULL_RESULT_SECONDS  # the batch becomes markers, freeing the body budget for the next
    assert len(operations.entries) == MAX_OPERATIONS

    document = render_canonical_json(
        {"nonces": nonces.to_json(), "operations": operations.to_json(), "retries": retries.to_json()},
        max_depth=6,
        max_length=4 << 20,
    )

    assert len(document) < 3 * 1024 * 1024  # under three quarters of the 4 MiB document ceiling
