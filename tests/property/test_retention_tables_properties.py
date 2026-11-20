"""Properties of the retention tables: cleanup never changes an answer, and encoding is an exact round trip."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from remote_energy_meter.protocol_v1 import encode_base64url
from remote_energy_meter.retention_tables import (
    FULL_RESULT_SECONDS,
    MARKER_SECONDS,
    ExactRetry,
    ExactRetryTable,
    NonceTable,
    OperationResult,
    OperationResultTable,
)

BASE = 1_900_000_000
_SPAN = FULL_RESULT_SECONDS + MARKER_SECONDS + 10_000

_times = st.integers(min_value=BASE - 5_000, max_value=BASE + _SPAN)
_small = st.integers(min_value=0, max_value=255)
_nonces = st.binary(min_size=16, max_size=16)
_digests = st.binary(min_size=32, max_size=32)


def _operation(n: int, committed_at: int, body: bool, digest: bytes) -> OperationResult:
    return OperationResult(
        operation_id=encode_base64url(bytes([n]) * 16, exact_length=16),
        digest=digest,
        status=200,
        body=b"body" if body else None,
        committed_at=committed_at,
    )


_operation_tables = st.lists(
    st.builds(_operation, _small, _times, st.booleans(), _digests), max_size=8, unique_by=lambda r: r.operation_id
).map(lambda entries: OperationResultTable(tuple(entries)))


@given(_operation_tables, _small, _digests, _times)
@settings(max_examples=300, deadline=None)
def test_compaction_never_changes_an_operation_answer(
    table: OperationResultTable, n: int, digest: bytes, now: int
) -> None:
    identifier = encode_base64url(bytes([n]) * 16, exact_length=16)

    assert table.lookup(identifier, digest, now) == table.compact(now).lookup(identifier, digest, now)


@given(_operation_tables)
@settings(max_examples=200, deadline=None)
def test_operation_tables_round_trip_exactly(table: OperationResultTable) -> None:
    assert OperationResultTable.from_json(table.to_json()) == table


_nonce_tables = st.lists(st.tuples(_nonces, _times), max_size=8, unique_by=lambda e: e[0]).map(
    lambda entries: NonceTable(tuple(entries))
)


@given(_nonce_tables.filter(lambda table: bool(table.entries)), st.data())
@settings(max_examples=300, deadline=None)
def test_compaction_never_changes_whether_a_retained_nonce_is_found(table: NonceTable, data: st.DataObject) -> None:
    """Query a nonce that is really in the table, at instants around its expiry, so both answers occur."""
    nonce, expires_at = data.draw(st.sampled_from(table.entries))
    now = expires_at + data.draw(st.integers(min_value=-2, max_value=2))

    assert table.contains(nonce, now) == table.compact(now).contains(nonce, now)
    assert table.contains(nonce, now) is (now <= expires_at)


@given(_nonce_tables, _nonces, _times)
@settings(max_examples=100, deadline=None)
def test_an_unknown_nonce_is_never_found(table: NonceTable, nonce: bytes, now: int) -> None:
    if nonce not in {held for held, _ in table.entries}:
        assert not table.contains(nonce, now)


@given(_nonce_tables)
@settings(max_examples=200, deadline=None)
def test_nonce_tables_round_trip_exactly(table: NonceTable) -> None:
    assert NonceTable.from_json(table.to_json()) == table


def _retry(n: int, expires_at: int, nonce: bytes, proposed: int | None) -> ExactRetry:
    return ExactRetry(
        purpose="rem-test-v1",
        request_hash=bytes([n]) * 32,
        nonce=nonce,
        key_generation=1,
        proposed_generation=proposed,
        status=200,
        body=b"body",
        expires_at=expires_at,
    )


_retry_tables = st.lists(
    st.builds(_retry, _small, _times, _nonces, st.one_of(st.none(), st.integers(0, 5))),
    max_size=6,
    unique_by=lambda e: e.request_hash,
).map(lambda entries: ExactRetryTable(tuple(entries)))


@given(_retry_tables.filter(lambda table: bool(table.entries)), st.data())
@settings(max_examples=300, deadline=None)
def test_compaction_never_changes_a_retry_answer_for_a_retained_request(
    table: ExactRetryTable, data: st.DataObject
) -> None:
    """Query the exact request of an entry in the table, around its expiry, so hits and misses both occur."""
    entry = data.draw(st.sampled_from(table.entries))
    now = entry.expires_at + data.draw(st.integers(min_value=-2, max_value=2))
    arguments = {
        "purpose": entry.purpose,
        "request_hash": entry.request_hash,
        "nonce": entry.nonce,
        "key_generation": entry.key_generation,
        "proposed_generation": entry.proposed_generation,
        "now": now,
    }

    answer = table.lookup(**arguments)  # type: ignore[arg-type]

    assert answer == table.compact(now).lookup(**arguments)  # type: ignore[arg-type]
    assert (answer is not None) is (now <= entry.expires_at)


@given(_retry_tables)
@settings(max_examples=200, deadline=None)
def test_retry_tables_round_trip_exactly(table: ExactRetryTable) -> None:
    assert ExactRetryTable.from_json(table.to_json()) == table
