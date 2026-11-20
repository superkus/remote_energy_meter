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


@given(_nonce_tables, _nonces, _times)
@settings(max_examples=300, deadline=None)
def test_compaction_never_changes_whether_a_nonce_is_retained(table: NonceTable, nonce: bytes, now: int) -> None:
    assert table.contains(nonce, now) == table.compact(now).contains(nonce, now)


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


@given(_retry_tables, _small, _nonces, _times)
@settings(max_examples=300, deadline=None)
def test_compaction_never_changes_a_retry_answer(table: ExactRetryTable, n: int, nonce: bytes, now: int) -> None:
    arguments = {
        "purpose": "rem-test-v1",
        "request_hash": bytes([n]) * 32,
        "nonce": nonce,
        "key_generation": 1,
        "proposed_generation": None,
        "now": now,
    }

    assert table.lookup(**arguments) == table.compact(now).lookup(**arguments)  # type: ignore[arg-type]


@given(_retry_tables)
@settings(max_examples=200, deadline=None)
def test_retry_tables_round_trip_exactly(table: ExactRetryTable) -> None:
    assert ExactRetryTable.from_json(table.to_json()) == table
