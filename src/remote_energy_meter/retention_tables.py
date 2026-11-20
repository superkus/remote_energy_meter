"""Pure, bounded retention tables for replay protection and exact retries.

The control and reporter profiles must remember three kinds of short-lived facts durably (design.md "Post-pair signed
control profile" and requirement 8.6d):

* :class:`NonceTable` -- every accepted signed-request nonce, kept through the request's expiry, so a replay is refused;
* :class:`ExactRetryTable` -- the exact status and body of a completed mutation, bound to the request that produced
  it, so a retry whose response was lost returns the original bytes without re-authorizing or repeating the action;
* :class:`OperationResultTable` -- the result of a roster action, kept in full for 24 hours and as a digest-only marker
  for the next 30 days, so a semantic retry returns the original result, then ``operation_result_expired``.

These are **values, not stores**. They do no file access and read no clock: every operation takes the current server
time as an argument and returns a new table. A domain store (the roster, the signed-control profile) embeds a table in
its own authenticated document, so that "nonce consumption, state change and retry result commit in one durable
transaction" (design.md) holds by construction: one document, one atomic write. The caller passes a time that already
went through its clock guard, so a clock that stepped back cannot reopen a window.

Boundaries differ on purpose and each is pinned by tests:

* a signed request is valid up to and including its ``expires`` instant ("not earlier than receipt", requirement 7.5f),
  so a nonce and a wire-retry record are kept while ``now <= expires_at`` and dropped after;
* an operation result is returnable while ``committed_at <= now < committed_at + 24 h``, is a marker while
  ``now < committed_at + 24 h + 30 d``, and is ignored from that instant (requirement 8.6d).

Lookups decide from the recorded instants only. Physical compaction (dropping bodies and old entries) runs on every
``record``/``consume`` but never changes an answer, so a lagging cleanup cannot return a result after its boundary.

Every table is capacity-bounded; a transaction that would exceed the bound is refused (:class:`TableRefusal`), never
evicting another entry. Bodies are plain bytes: **never record a response that contains key material** here; the MQTT
key-provisioning response has its own encrypted record (task 7.2.1).

Each table encodes to and decodes from plain JSON values with strict validation (``ValueError`` on anything else), so a
domain store can fold them into its record and treat a decode failure as a malformed record.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Final, TypeGuard

from remote_energy_meter.protocol_v1 import JsonValue, ProtocolViolation, decode_base64url, encode_base64url
from remote_energy_meter.protocol_v1.encoding import decode_opaque_base64url

__all__ = [
    "FULL_RESULT_SECONDS",
    "MARKER_SECONDS",
    "MAX_BODY_BYTES",
    "MAX_NONCES",
    "MAX_OPERATIONS",
    "MAX_RETRIES",
    "ExactRetry",
    "ExactRetryTable",
    "NonceTable",
    "OperationLookup",
    "OperationResult",
    "OperationResultTable",
    "OperationState",
    "TableRefusal",
    "TableRefused",
]

FULL_RESULT_SECONDS: Final = 24 * 60 * 60
"""An operation's full result is returnable for this long after it commits."""
MARKER_SECONDS: Final = 30 * 24 * 60 * 60
"""After the full window, a digest-only marker answers retries for this long, then the operation is ignored."""
MAX_NONCES: Final = 1024
MAX_RETRIES: Final = 256
MAX_OPERATIONS: Final = 4096
MAX_BODY_BYTES: Final = 2048
"""Cap on a recorded response body."""

_UINT64_MAX: Final = 2**64 - 1
_NONCE_BYTES: Final = 16
_DIGEST_BYTES: Final = 32
_PURPOSE_MAX: Final = 64


class TableRefusal(StrEnum):
    """Why a table refused a change. A refusal changes nothing."""

    EXPIRED = "expired"
    REPLAYED = "replayed"
    EXISTS = "exists"
    CAPACITY_EXCEEDED = "capacity_exceeded"


class TableRefused(Exception):  # noqa: N818 - a refusal, not an error: the table is simply unchanged
    """A requested change violates a table rule; the table is unchanged. ``reason`` is the rule."""

    def __init__(self, reason: TableRefusal, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


def _is_uint64(value: object) -> TypeGuard[int]:
    return type(value) is int and 0 <= value <= _UINT64_MAX


def _check_instant(value: int, name: str) -> None:
    if not _is_uint64(value):
        raise ValueError(f"{name} must be a uint64 Unix time")


def _check_bytes(value: bytes, length: int, name: str) -> None:
    if type(value) is not bytes or len(value) != length:
        raise ValueError(f"{name} must be exactly {length} bytes")


def _check_body(body: bytes) -> None:
    if type(body) is not bytes or not 1 <= len(body) <= MAX_BODY_BYTES:
        raise ValueError(f"a recorded body is between 1 and {MAX_BODY_BYTES} bytes")


def _check_status(status: int) -> None:
    if type(status) is not int or not 100 <= status <= 599:
        raise ValueError("status must be an HTTP status code")


def _check_identifier(value: str, name: str) -> None:
    try:
        decode_base64url(value, exact_length=16)
    except ProtocolViolation:
        raise ValueError(f"{name} must be a canonical 22-character base64url value") from None


def _check_purpose(value: str) -> None:
    if (
        type(value) is not str
        or not 1 <= len(value) <= _PURPOSE_MAX
        or not value.isascii()
        or not all(c.islower() or c.isdigit() or c == "-" for c in value)
    ):
        raise ValueError("purpose must be 1 to 64 lowercase ASCII letters, digits or hyphens")


def _expect_object(raw: JsonValue, keys: frozenset[str]) -> dict[str, JsonValue]:
    if type(raw) is not dict or set(raw) != keys:
        raise ValueError("unexpected record shape")
    return raw


def _expect_list(raw: JsonValue, limit: int) -> list[JsonValue]:
    if type(raw) is not list or len(raw) > limit:
        raise ValueError("unexpected list")
    return raw


# --------------------------------------------------------------------------------------------------
# replay nonces
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NonceTable:
    """Accepted signed-request nonces, each with the instant its request stops being valid.

    A nonce is retained while ``now <= expires_at`` (a request is valid up to and including its ``expires``).
    """

    entries: tuple[tuple[bytes, int], ...] = ()

    def consume(self, nonce: bytes, expires_at: int, now: int) -> NonceTable:
        """Accept ``nonce`` once.

        Raises:
            ValueError: An argument is malformed.
            TableRefused: ``EXPIRED`` (the request is no longer valid at ``now``), ``REPLAYED`` (the nonce is retained
                and unexpired), or ``CAPACITY_EXCEEDED`` (too many unexpired nonces; nothing is evicted).
        """
        _check_bytes(nonce, _NONCE_BYTES, "nonce")
        _check_instant(expires_at, "expires_at")
        _check_instant(now, "now")
        if expires_at < now:
            raise TableRefused(TableRefusal.EXPIRED, "the request has expired")
        live = self.compact(now).entries
        if any(held == nonce for held, _ in live):
            raise TableRefused(TableRefusal.REPLAYED, "that nonce was already accepted")
        if len(live) >= MAX_NONCES:
            raise TableRefused(TableRefusal.CAPACITY_EXCEEDED, "too many unexpired nonces")
        return NonceTable((*live, (nonce, expires_at)))

    def contains(self, nonce: bytes, now: int) -> bool:
        """Whether ``nonce`` is retained and unexpired at ``now``."""
        return any(held == nonce and now <= expires_at for held, expires_at in self.entries)

    def compact(self, now: int) -> NonceTable:
        """Drop nonces whose request is no longer valid at ``now``."""
        return NonceTable(tuple(entry for entry in self.entries if now <= entry[1]))

    def to_json(self) -> list[JsonValue]:
        """Encode as plain JSON values."""
        return [
            {"expires_at": expires_at, "nonce": encode_base64url(nonce, exact_length=_NONCE_BYTES)}
            for nonce, expires_at in self.entries
        ]

    @classmethod
    def from_json(cls, raw: JsonValue) -> NonceTable:
        """Decode; ``ValueError`` for anything but the exact encoding of a valid table."""
        entries: list[tuple[bytes, int]] = []
        for item in _expect_list(raw, MAX_NONCES):
            fields = _expect_object(item, frozenset({"expires_at", "nonce"}))
            nonce, expires_at = fields["nonce"], fields["expires_at"]
            if type(nonce) is not str or not _is_uint64(expires_at):
                raise ValueError("nonce entry")
            entries.append((decode_base64url(nonce, exact_length=_NONCE_BYTES), expires_at))
        if len({nonce for nonce, _ in entries}) != len(entries):
            raise ValueError("a nonce is recorded twice")
        return cls(tuple(entries))


# --------------------------------------------------------------------------------------------------
# exact wire retries
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExactRetry:
    """A completed mutation's exact result, bound to the request that produced it."""

    purpose: str
    request_hash: bytes
    nonce: bytes
    key_generation: int
    proposed_generation: int | None
    status: int
    body: bytes = field(repr=False)
    expires_at: int


@dataclass(frozen=True, slots=True)
class ExactRetryTable:
    """Exact-retry records, returned only for an unexpired request that matches every bound field.

    A record is returnable while ``now <= expires_at``, the instant the original signed request stops being valid.
    """

    entries: tuple[ExactRetry, ...] = ()

    def lookup(
        self,
        *,
        purpose: str,
        request_hash: bytes,
        nonce: bytes,
        key_generation: int,
        proposed_generation: int | None,
        now: int,
    ) -> ExactRetry | None:
        """Return the record that matches every bound field and is unexpired at ``now``, else ``None``."""
        for entry in self.entries:
            if (
                now <= entry.expires_at
                and entry.purpose == purpose
                and entry.request_hash == request_hash
                and entry.nonce == nonce
                and entry.key_generation == key_generation
                and entry.proposed_generation == proposed_generation
            ):
                return entry
        return None

    def record(self, entry: ExactRetry, now: int) -> ExactRetryTable:
        """Add ``entry``.

        Raises:
            ValueError: A field is malformed.
            TableRefused: ``EXPIRED`` (``expires_at`` is before ``now``), ``EXISTS`` (an unexpired record for the same
                purpose and request hash exists) or ``CAPACITY_EXCEEDED``.
        """
        _check_purpose(entry.purpose)
        _check_bytes(entry.request_hash, _DIGEST_BYTES, "request_hash")
        _check_bytes(entry.nonce, _NONCE_BYTES, "nonce")
        _check_instant(entry.key_generation, "key_generation")
        if entry.proposed_generation is not None:
            _check_instant(entry.proposed_generation, "proposed_generation")
        _check_status(entry.status)
        _check_body(entry.body)
        _check_instant(entry.expires_at, "expires_at")
        _check_instant(now, "now")
        if entry.expires_at < now:
            raise TableRefused(TableRefusal.EXPIRED, "the request has expired")
        live = self.compact(now).entries
        if any(held.purpose == entry.purpose and held.request_hash == entry.request_hash for held in live):
            raise TableRefused(TableRefusal.EXISTS, "a result is already recorded for that request")
        if len(live) >= MAX_RETRIES:
            raise TableRefused(TableRefusal.CAPACITY_EXCEEDED, "too many unexpired recorded retries")
        return ExactRetryTable((*live, entry))

    def compact(self, now: int) -> ExactRetryTable:
        """Drop records whose request is no longer valid at ``now``."""
        return ExactRetryTable(tuple(entry for entry in self.entries if now <= entry.expires_at))

    def to_json(self) -> list[JsonValue]:
        """Encode as plain JSON values."""
        return [
            {
                "body": encode_base64url(e.body, exact_length=len(e.body)),
                "expires_at": e.expires_at,
                "key_generation": e.key_generation,
                "nonce": encode_base64url(e.nonce, exact_length=_NONCE_BYTES),
                "proposed_generation": e.proposed_generation,
                "purpose": e.purpose,
                "request_hash": encode_base64url(e.request_hash, exact_length=_DIGEST_BYTES),
                "status": e.status,
            }
            for e in self.entries
        ]

    @classmethod
    def from_json(cls, raw: JsonValue) -> ExactRetryTable:
        """Decode; ``ValueError`` for anything but the exact encoding of a valid table."""
        keys = frozenset(
            {
                "body",
                "expires_at",
                "key_generation",
                "nonce",
                "proposed_generation",
                "purpose",
                "request_hash",
                "status",
            }
        )
        entries: list[ExactRetry] = []
        for item in _expect_list(raw, MAX_RETRIES):
            f = _expect_object(item, keys)
            body, nonce, purpose, request_hash = f["body"], f["nonce"], f["purpose"], f["request_hash"]
            proposed, status = f["proposed_generation"], f["status"]
            key_generation, expires_at = f["key_generation"], f["expires_at"]
            if (
                type(body) is not str
                or type(nonce) is not str
                or type(purpose) is not str
                or type(request_hash) is not str
                or not (proposed is None or _is_uint64(proposed))
                or not _is_uint64(key_generation)
                or not _is_uint64(expires_at)
                or type(status) is not int
            ):
                raise ValueError("retry entry")
            entry = ExactRetry(
                purpose=purpose,
                request_hash=decode_base64url(request_hash, exact_length=_DIGEST_BYTES),
                nonce=decode_base64url(nonce, exact_length=_NONCE_BYTES),
                key_generation=key_generation,
                proposed_generation=proposed,
                status=status,
                body=decode_opaque_base64url(body, minimum=1, maximum=MAX_BODY_BYTES),
                expires_at=expires_at,
            )
            _check_purpose(entry.purpose)
            _check_status(entry.status)
            entries.append(entry)
        if len({(e.purpose, e.request_hash) for e in entries}) != len(entries):
            raise ValueError("a request is recorded twice")
        return cls(tuple(entries))


# --------------------------------------------------------------------------------------------------
# roster action results
# --------------------------------------------------------------------------------------------------


class OperationState(StrEnum):
    """Where an operation result is at a given instant."""

    FULL = "full"
    MARKER = "marker"
    GONE = "gone"


@dataclass(frozen=True, slots=True)
class OperationResult:
    """The recorded result of one roster action. ``body`` is ``None`` once compacted to a marker."""

    operation_id: str
    digest: bytes
    status: int
    body: bytes | None = field(repr=False)
    committed_at: int

    def state(self, now: int) -> OperationState:
        """The window ``now`` falls in, decided only from ``committed_at``."""
        if now < self.committed_at + FULL_RESULT_SECONDS:
            return OperationState.FULL
        if now < self.committed_at + FULL_RESULT_SECONDS + MARKER_SECONDS:
            return OperationState.MARKER
        return OperationState.GONE


@dataclass(frozen=True, slots=True)
class OperationLookup:
    """The answer to an operation retry.

    ``state`` is ``GONE`` when nothing applies (the caller proceeds with ordinary rules). ``conflict`` is true when
    the identifier is known (full or marker) under a different digest, which the caller reports as ``409``.
    ``status`` and ``body`` are set only for a full-window exact match; a marker match has ``state`` ``MARKER``.
    """

    state: OperationState
    conflict: bool = False
    status: int | None = None
    body: bytes | None = None


@dataclass(frozen=True, slots=True)
class OperationResultTable:
    """Roster action results: full for 24 hours, then a digest-only marker for 30 days, then ignored."""

    entries: tuple[OperationResult, ...] = ()

    def lookup(self, operation_id: str, digest: bytes, now: int) -> OperationLookup:
        """Answer a retry from the recorded instants only, whether or not cleanup has run."""
        for entry in self.entries:
            if entry.operation_id != operation_id:
                continue
            state = entry.state(now)
            if state is OperationState.GONE:
                continue
            if entry.digest != digest:
                return OperationLookup(state, conflict=True)
            if state is OperationState.FULL and entry.body is not None:
                return OperationLookup(state, status=entry.status, body=entry.body)
            return OperationLookup(OperationState.MARKER if state is OperationState.FULL else state)
        return OperationLookup(OperationState.GONE)

    def record(self, result: OperationResult, now: int) -> OperationResultTable:
        """Record a committed action's result.

        Raises:
            ValueError: A field is malformed, or ``committed_at`` is not ``now``.
            TableRefused: ``EXISTS`` (the identifier is still known, full or marker) or ``CAPACITY_EXCEEDED``.
        """
        _check_identifier(result.operation_id, "operation_id")
        _check_bytes(result.digest, _DIGEST_BYTES, "digest")
        _check_status(result.status)
        if result.body is None:
            raise ValueError("a new result carries its body")
        _check_body(result.body)
        _check_instant(result.committed_at, "committed_at")
        _check_instant(now, "now")
        if result.committed_at != now:
            raise ValueError("a result commits at the time it is recorded")
        live = self.compact(now).entries
        if any(entry.operation_id == result.operation_id for entry in live):
            raise TableRefused(TableRefusal.EXISTS, "that operation identifier is already recorded")
        if len(live) >= MAX_OPERATIONS:
            raise TableRefused(TableRefusal.CAPACITY_EXCEEDED, "too many retained operation results")
        return OperationResultTable((*live, result))

    def compact(self, now: int) -> OperationResultTable:
        """Delete bodies at the 24-hour boundary and entries at the end of the marker window."""
        kept = tuple(
            entry if entry.state(now) is OperationState.FULL else replace(entry, body=None)
            for entry in self.entries
            if entry.state(now) is not OperationState.GONE
        )
        return OperationResultTable(kept)

    def to_json(self) -> list[JsonValue]:
        """Encode as plain JSON values."""
        return [
            {
                "body": None if e.body is None else encode_base64url(e.body, exact_length=len(e.body)),
                "committed_at": e.committed_at,
                "digest": encode_base64url(e.digest, exact_length=_DIGEST_BYTES),
                "id": e.operation_id,
                "status": e.status,
            }
            for e in self.entries
        ]

    @classmethod
    def from_json(cls, raw: JsonValue) -> OperationResultTable:
        """Decode; ``ValueError`` for anything but the exact encoding of a valid table."""
        keys = frozenset({"body", "committed_at", "digest", "id", "status"})
        entries: list[OperationResult] = []
        for item in _expect_list(raw, MAX_OPERATIONS):
            f = _expect_object(item, keys)
            body, identifier, digest, status = f["body"], f["id"], f["digest"], f["status"]
            committed_at = f["committed_at"]
            if (
                not (body is None or type(body) is str)
                or type(identifier) is not str
                or type(digest) is not str
                or type(status) is not int
                or not _is_uint64(committed_at)
            ):
                raise ValueError("operation entry")
            _check_identifier(identifier, "id")
            _check_status(status)
            entries.append(
                OperationResult(
                    operation_id=identifier,
                    digest=decode_base64url(digest, exact_length=_DIGEST_BYTES),
                    status=status,
                    body=None if body is None else decode_opaque_base64url(body, minimum=1, maximum=MAX_BODY_BYTES),
                    committed_at=committed_at,
                )
            )
        if len({entry.operation_id for entry in entries}) != len(entries):
            raise ValueError("an operation identifier is recorded twice")
        return cls(tuple(entries))
