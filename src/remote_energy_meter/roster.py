"""The durable reporter roster: who is enrolled, under which key, and what happened to them.

Requirements 2.2, 3.1, 8.1, 8.6b and 8.9. One authenticated document holds the enrolled reporters, the deletion
tombstones, and the ``roster_generation`` cursor, so a reporter change, the new cursor value and the event it produces
are always written together.

**Identity.** A reporter's ID is the RFC 7638 thumbprint of its *initial* enrollment key and never changes. The key it
signs with can be rotated (:meth:`RosterStore.rotate_key`); each rotation advances ``key_generation`` by exactly one.
Every *current* key is unique across the roster, so a signature identifies one reporter. A key a reporter has rotated
away from is not reserved: it stops identifying anyone, and the roster keeps no history of retired keys. A reporter
that lost or may have lost its key is revoked and re-enrolled; there is no unauthenticated recovery.

**Lifecycle.** :meth:`~RosterStore.activate` (reporter-confirmed activation) creates an ``active`` record. A
reporter is ``active`` until :meth:`~RosterStore.revoke` makes it ``revoked``, which keeps the record, so its device,
entities and energy history stay. :meth:`~RosterStore.delete` removes it and leaves a **tombstone**; at most 1,000
tombstones (the pending deletion-cleanups of requirement 8.9) may exist, and a delete past that is refused without
changing anything. :meth:`~RosterStore.compact_tombstone` removes one when its cleanup is complete. At most 1,000
reporters are enrolled (active or revoked, requirement 8.1).

**The cursor.** The cursor starts at zero and rises by exactly one for each control-visible mutation (requirement
8.6b). Pending enrollment and pending key rotation are memory-only, but each of their events also advances the durable
cursor (:meth:`~RosterStore.record_transient_event`), so a restart that loses the pending payloads is detected as a
cursor ahead of any event the integration saw, and forces snapshot reconciliation. A refused change, an exact retry
and a metadata update that changes nothing do not advance it. Each mutation returns the :class:`RosterChange` it
produced (generation, event type, reporter, reporter state generation, and the complete current record or the
tombstone), which is exactly the immutable event of requirement 8.6b.

Every mutation takes an optional ``expected_roster_generation``; when given and not current, the change is refused as
``STALE_GENERATION`` (the caller answers ``409``), atomically with the write.

Nothing here verifies signatures or administrator authorization, holds sessions, or retains an event history; the
control routes do the first two (task 7.2) and the event stream and snapshots serve history (tasks 7.2.2, 7.2.3). The
control actions' operation results and replay nonces are embedded in this same document by task 7.2.4, so that an
action, its result and its nonce commit together.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from remote_energy_meter.authenticated_store import AuthenticatedDocumentStore, StateAuthenticationKeyStore
from remote_energy_meter.key_material import require_aware
from remote_energy_meter.protocol_v1 import (
    JsonValue,
    ProtocolViolation,
    decode_base64url,
    parse_p256_public_jwk,
)
from remote_energy_meter.reporter_metadata import ReporterMetadata
from remote_energy_meter.secure_store import SecureStoreError, StoredDocument, StoreFault, ensure_private_directory

__all__ = [
    "DOCUMENT_NAME",
    "MAX_REPORTERS",
    "MAX_TOMBSTONES",
    "TRANSIENT_EVENTS",
    "ReporterRecord",
    "ReporterState",
    "Roster",
    "RosterChange",
    "RosterEvent",
    "RosterRefusal",
    "RosterRefused",
    "RosterStore",
    "Tombstone",
]

DOCUMENT_NAME: Final = "reporter-roster.json"
MAX_REPORTERS: Final = 1000
"""Enrolled reporters, active or revoked (requirement 8.1)."""
MAX_TOMBSTONES: Final = 1000
"""Pending deletion-cleanup records (requirement 8.9)."""

_UINT64_MAX: Final = 2**64 - 1
_STORE_ID: Final = "reporter-roster"
_SCHEMA_VERSION: Final = 1
_MAX_BYTES: Final = 2 * 1024 * 1024
_RECORD_KEYS: Final = frozenset({"generation", "reporters", "tombstones"})
_REPORTER_KEYS: Final = frozenset(
    {
        "enrolled_at",
        "jwk",
        "key_generation",
        "key_thumbprint",
        "metadata",
        "reporter_id",
        "revoked_at",
        "state",
        "state_generation",
    }
)
_TOMBSTONE_KEYS: Final = frozenset({"deleted_at_generation", "reporter_id", "state_generation"})


class ReporterState(StrEnum):
    """A stored reporter is active, or revoked with its record kept. A deleted reporter has only a tombstone."""

    ACTIVE = "active"
    REVOKED = "revoked"


class RosterEvent(StrEnum):
    """The closed set of control-visible roster events (requirement 8.6b)."""

    REPORTER_ACTIVATED = "reporter_activated"
    METADATA_CHANGED = "metadata_changed"
    KEY_ROTATED = "key_rotated"
    REPORTER_REVOKED = "reporter_revoked"
    REPORTER_DELETED = "reporter_deleted"
    PENDING_CREATED = "pending_created"
    PENDING_APPROVED = "pending_approved"
    PENDING_REJECTED = "pending_rejected"
    PENDING_EXPIRED = "pending_expired"
    ROTATION_PENDING = "rotation_pending"
    ROTATION_APPROVED = "rotation_approved"
    ROTATION_EXPIRED = "rotation_expired"


TRANSIENT_EVENTS: Final = frozenset(
    {
        RosterEvent.PENDING_CREATED,
        RosterEvent.PENDING_APPROVED,
        RosterEvent.PENDING_REJECTED,
        RosterEvent.PENDING_EXPIRED,
        RosterEvent.ROTATION_PENDING,
        RosterEvent.ROTATION_APPROVED,
        RosterEvent.ROTATION_EXPIRED,
    }
)
"""Events whose payload is memory-only: they advance the durable cursor but change no stored record."""


class RosterRefusal(StrEnum):
    """Why a change was refused. A refusal changes nothing."""

    NOT_FOUND = "not_found"
    EXISTS = "exists"
    TOMBSTONED = "tombstoned"
    NOT_ACTIVE = "not_active"
    KEY_IN_USE = "key_in_use"
    KEY_GENERATION_MISMATCH = "key_generation_mismatch"
    STALE_GENERATION = "stale_generation"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    GENERATION_EXHAUSTED = "generation_exhausted"


class RosterRefused(Exception):  # noqa: N818 - a refusal, not an error: the roster is simply unchanged
    """A requested change violates a roster rule; nothing was written. ``reason`` is the rule."""

    def __init__(self, reason: RosterRefusal, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


class _NoChange(Exception):  # noqa: N818 - internal signal that a request changes nothing, so nothing is written
    pass


@dataclass(frozen=True, slots=True)
class ReporterRecord:
    """One enrolled reporter."""

    reporter_id: str
    state: ReporterState
    jwk: bytes
    key_thumbprint: str
    key_generation: int
    state_generation: int
    metadata: ReporterMetadata
    enrolled_at: int
    revoked_at: int | None


@dataclass(frozen=True, slots=True)
class Tombstone:
    """What remains of a deleted reporter until its cleanup completes."""

    reporter_id: str
    state_generation: int
    deleted_at_generation: int


@dataclass(frozen=True, slots=True)
class RosterChange:
    """The immutable event one mutation produced (requirement 8.6b)."""

    generation: int
    event: RosterEvent
    reporter_id: str
    reporter_state_generation: int
    record: ReporterRecord | None = None
    tombstone: Tombstone | None = None


@dataclass(frozen=True, slots=True)
class Roster:
    """A verified snapshot of the roster."""

    generation: int
    reporters: tuple[ReporterRecord, ...]
    tombstones: tuple[Tombstone, ...]

    def get(self, reporter_id: str) -> ReporterRecord | None:
        """The stored record for ``reporter_id``, or ``None``."""
        return next((r for r in self.reporters if r.reporter_id == reporter_id), None)

    def tombstone(self, reporter_id: str) -> Tombstone | None:
        """The tombstone for ``reporter_id``, or ``None``."""
        return next((t for t in self.tombstones if t.reporter_id == reporter_id), None)

    def reporter_for_key(self, key_thumbprint: str) -> ReporterRecord | None:
        """The reporter whose *current* signing key has this thumbprint, or ``None``."""
        return next((r for r in self.reporters if r.key_thumbprint == key_thumbprint), None)


_EMPTY: Final = Roster(0, (), ())


def _is_uint64(value: object) -> bool:
    return type(value) is int and 0 <= value <= _UINT64_MAX


def _check_reporter_id(value: str) -> None:
    try:
        decode_base64url(value, exact_length=32)
    except ProtocolViolation:
        raise ValueError("a reporter ID is a canonical 43-character base64url RFC 7638 thumbprint") from None


class RosterStore:
    """Durable, authenticated storage of the reporter roster.

    Args:
        directory: A dedicated owner-only directory for this document.
        keys: The state-authentication key store whose key tags the record.
        clock: Returns the current time as a timezone-aware ``datetime``; used for the informational enrollment and
            revocation times only, never for a rule.
    """

    def __init__(
        self,
        directory: Path,
        *,
        keys: StateAuthenticationKeyStore,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._clock = clock
        self._directory = directory
        self._document = AuthenticatedDocumentStore(
            directory,
            DOCUMENT_NAME,
            store_id=_STORE_ID,
            schema_version=_SCHEMA_VERSION,
            max_bytes=_MAX_BYTES,
            keys=keys,
        )

    @property
    def path(self) -> Path:
        """The document's path."""
        return self._document.path

    def create(self) -> Roster:
        """Write the initial empty roster, once.

        Raises:
            SecureStoreError: ``IDENTITY_EXISTS`` when a roster already exists (it is never replaced), including one
                created while this call runs; any other fault of the authenticated store.
        """
        ensure_private_directory(self._directory)
        if self._document.load() is not None:
            raise SecureStoreError(StoreFault.IDENTITY_EXISTS, f"{self.path} already exists and is never replaced")
        try:
            self._document.create(self._encode(_EMPTY))
        except SecureStoreError as error:
            if error.fault is StoreFault.CONCURRENT_MODIFICATION:
                detail = f"{self.path} was created by another writer and is never replaced"
                raise SecureStoreError(StoreFault.IDENTITY_EXISTS, detail) from None
            raise
        return _EMPTY

    def load(self) -> Roster:
        """Read and fully verify the roster.

        Raises:
            SecureStoreError: ``IDENTITY_MISSING`` when no roster exists; ``MALFORMED`` or ``IDENTITY_INCONSISTENT``
                for a record that breaks its rules; any fault of the authenticated store.
        """
        document = self._document.load()
        if document is None:
            raise SecureStoreError(StoreFault.IDENTITY_MISSING, f"{self.path} does not exist")
        return self._decode(document.record)

    # -- lifecycle --------------------------------------------------------------------------------

    def activate(
        self, jwk: bytes, metadata: ReporterMetadata, *, expected_roster_generation: int | None = None
    ) -> RosterChange:
        """Reporter-confirmed activation: enroll a reporter under the thumbprint of its initial key.

        Raises:
            ValueError: ``jwk`` is not an exact canonical public P-256 JWK.
            RosterRefused: ``EXISTS`` (that ID is enrolled), ``TOMBSTONED`` (that ID still has a deletion cleanup
                pending), ``KEY_IN_USE`` (the key is another reporter's current key), ``CAPACITY_EXCEEDED``,
                ``STALE_GENERATION`` or ``GENERATION_EXHAUSTED``.
            SecureStoreError: Any fault of the authenticated store.
        """
        parsed = parse_p256_public_jwk(jwk)
        now = self._now()
        reporter_id = parsed.thumbprint

        def change(roster: Roster, generation: int) -> tuple[Roster, RosterChange]:
            if roster.get(reporter_id) is not None:
                raise RosterRefused(RosterRefusal.EXISTS, "that reporter is already enrolled")
            if roster.tombstone(reporter_id) is not None:
                raise RosterRefused(RosterRefusal.TOMBSTONED, "that reporter's deletion cleanup is still pending")
            if roster.reporter_for_key(reporter_id) is not None:
                raise RosterRefused(RosterRefusal.KEY_IN_USE, "that key is another reporter's current key")
            if len(roster.reporters) >= MAX_REPORTERS:
                raise RosterRefused(RosterRefusal.CAPACITY_EXCEEDED, "the roster is full")
            record = ReporterRecord(
                reporter_id=reporter_id,
                state=ReporterState.ACTIVE,
                jwk=parsed.canonical_bytes,
                key_thumbprint=parsed.thumbprint,
                key_generation=1,
                state_generation=1,
                metadata=metadata,
                enrolled_at=now,
                revoked_at=None,
            )
            reporters = tuple(sorted((*roster.reporters, record), key=lambda r: r.reporter_id))
            changed = replace(roster, generation=generation, reporters=reporters)
            return changed, RosterChange(generation, RosterEvent.REPORTER_ACTIVATED, reporter_id, 1, record)

        return self._apply(change, expected_roster_generation)

    def update_metadata(
        self, reporter_id: str, metadata: ReporterMetadata, *, expected_roster_generation: int | None = None
    ) -> RosterChange | None:
        """Change an active reporter's display metadata. Returns ``None`` and writes nothing if nothing changes.

        Raises:
            ValueError: ``reporter_id`` is malformed.
            RosterRefused: ``NOT_FOUND``, ``NOT_ACTIVE``, ``STALE_GENERATION`` or ``GENERATION_EXHAUSTED``.
            SecureStoreError: Any fault of the authenticated store.
        """
        _check_reporter_id(reporter_id)

        def change(roster: Roster, generation: int) -> tuple[Roster, RosterChange]:
            record = _require_active(roster, reporter_id)
            if record.metadata == metadata:
                raise _NoChange
            updated = replace(record, metadata=metadata, state_generation=record.state_generation + 1)
            return _replace_record(roster, updated, generation), RosterChange(
                generation, RosterEvent.METADATA_CHANGED, reporter_id, updated.state_generation, updated
            )

        try:
            return self._apply(change, expected_roster_generation)
        except _NoChange:
            return None

    def rotate_key(
        self,
        reporter_id: str,
        *,
        current_generation: int,
        proposed_jwk: bytes,
        expected_roster_generation: int | None = None,
    ) -> RosterChange:
        """Swap an active reporter's signing key; its ID does not change.

        Args:
            reporter_id: The reporter's stable ID.
            current_generation: The key generation the caller believes is current; the next one is this plus one.
            proposed_jwk: The new key's exact canonical public JWK.

        Raises:
            ValueError: ``reporter_id`` or ``proposed_jwk`` is malformed.
            RosterRefused: ``NOT_FOUND``, ``NOT_ACTIVE``, ``KEY_GENERATION_MISMATCH``, ``KEY_IN_USE`` (the proposed key
                is any reporter's current key, or an enrolled or deleted reporter ID), ``STALE_GENERATION`` or
                ``GENERATION_EXHAUSTED``.
            SecureStoreError: Any fault of the authenticated store.
        """
        _check_reporter_id(reporter_id)
        if not _is_uint64(current_generation):
            raise ValueError("current_generation must be an unsigned 64-bit integer")
        proposed = parse_p256_public_jwk(proposed_jwk)

        def change(roster: Roster, generation: int) -> tuple[Roster, RosterChange]:
            record = _require_active(roster, reporter_id)
            if record.key_generation != current_generation:
                raise RosterRefused(
                    RosterRefusal.KEY_GENERATION_MISMATCH, f"the current key generation is {record.key_generation}"
                )
            # No overflow check: key_generation <= state_generation <= roster generation, which is below its maximum
            # here (``_apply`` refuses an exhausted cursor first), so the increment below cannot pass 2**64 - 1.
            if (
                roster.reporter_for_key(proposed.thumbprint) is not None
                or roster.get(proposed.thumbprint) is not None
                or roster.tombstone(proposed.thumbprint) is not None
            ):
                # A key is also never reused as another identity: not a current key, not an enrolled or deleted ID.
                raise RosterRefused(RosterRefusal.KEY_IN_USE, "that key is already in use by the roster")
            updated = replace(
                record,
                jwk=proposed.canonical_bytes,
                key_thumbprint=proposed.thumbprint,
                key_generation=record.key_generation + 1,
                state_generation=record.state_generation + 1,
            )
            return _replace_record(roster, updated, generation), RosterChange(
                generation, RosterEvent.KEY_ROTATED, reporter_id, updated.state_generation, updated
            )

        return self._apply(change, expected_roster_generation)

    def revoke(self, reporter_id: str, *, expected_roster_generation: int | None = None) -> RosterChange:
        """Revoke an active reporter. Its record, device and history are kept; its key stops being accepted.

        Raises:
            ValueError: ``reporter_id`` is malformed.
            RosterRefused: ``NOT_FOUND``, ``NOT_ACTIVE`` (already revoked), ``STALE_GENERATION`` or
                ``GENERATION_EXHAUSTED``.
            SecureStoreError: Any fault of the authenticated store.
        """
        _check_reporter_id(reporter_id)
        now = self._now()

        def change(roster: Roster, generation: int) -> tuple[Roster, RosterChange]:
            record = _require_active(roster, reporter_id)
            updated = replace(
                record, state=ReporterState.REVOKED, state_generation=record.state_generation + 1, revoked_at=now
            )
            return _replace_record(roster, updated, generation), RosterChange(
                generation, RosterEvent.REPORTER_REVOKED, reporter_id, updated.state_generation, updated
            )

        return self._apply(change, expected_roster_generation)

    def delete(self, reporter_id: str, *, expected_roster_generation: int | None = None) -> RosterChange:
        """Delete a reporter (active or revoked), leaving a tombstone until its cleanup completes.

        A delete is refused, changing nothing, when 1,000 deletion cleanups are already pending.

        Raises:
            ValueError: ``reporter_id`` is malformed.
            RosterRefused: ``NOT_FOUND``, ``CAPACITY_EXCEEDED``, ``STALE_GENERATION`` or ``GENERATION_EXHAUSTED``.
            SecureStoreError: Any fault of the authenticated store.
        """
        _check_reporter_id(reporter_id)

        def change(roster: Roster, generation: int) -> tuple[Roster, RosterChange]:
            record = roster.get(reporter_id)
            if record is None:
                raise RosterRefused(RosterRefusal.NOT_FOUND, "no such reporter")
            if len(roster.tombstones) >= MAX_TOMBSTONES:
                raise RosterRefused(RosterRefusal.CAPACITY_EXCEEDED, "too many deletion cleanups are pending")
            tombstone = Tombstone(reporter_id, record.state_generation + 1, generation)
            reporters = tuple(r for r in roster.reporters if r.reporter_id != reporter_id)
            tombstones = tuple(sorted((*roster.tombstones, tombstone), key=lambda t: t.reporter_id))
            changed = replace(roster, generation=generation, reporters=reporters, tombstones=tombstones)
            return changed, RosterChange(
                generation, RosterEvent.REPORTER_DELETED, reporter_id, tombstone.state_generation, None, tombstone
            )

        return self._apply(change, expected_roster_generation)

    def record_transient_event(
        self,
        event: RosterEvent,
        reporter_id: str,
        reporter_state_generation: int,
        *,
        expected_roster_generation: int | None = None,
    ) -> RosterChange:
        """Advance the durable cursor for an event whose payload is memory-only (a pending enrollment or rotation).

        Raises:
            ValueError: ``event`` is not one of :data:`TRANSIENT_EVENTS`, or an argument is malformed.
            RosterRefused: ``STALE_GENERATION`` or ``GENERATION_EXHAUSTED``.
            SecureStoreError: Any fault of the authenticated store.
        """
        if event not in TRANSIENT_EVENTS:
            raise ValueError(f"{event!r} is not a memory-only roster event")
        _check_reporter_id(reporter_id)
        if not _is_uint64(reporter_state_generation):
            raise ValueError("reporter_state_generation must be an unsigned 64-bit integer")

        def change(roster: Roster, generation: int) -> tuple[Roster, RosterChange]:
            return replace(roster, generation=generation), RosterChange(
                generation, event, reporter_id, reporter_state_generation
            )

        return self._apply(change, expected_roster_generation)

    def compact_tombstone(self, reporter_id: str) -> Roster:
        """Remove a tombstone whose cleanup is complete. This is internal bookkeeping, not a control-visible event, so
        the cursor does not advance.

        Raises:
            ValueError: ``reporter_id`` is malformed.
            RosterRefused: ``NOT_FOUND`` when there is no such tombstone.
            SecureStoreError: Any fault of the authenticated store.
        """
        _check_reporter_id(reporter_id)

        def build(current: StoredDocument | None) -> Mapping[str, JsonValue]:
            roster = self._require(current)
            if roster.tombstone(reporter_id) is None:
                raise RosterRefused(RosterRefusal.NOT_FOUND, "no such tombstone")
            kept = tuple(t for t in roster.tombstones if t.reporter_id != reporter_id)
            return self._encode(replace(roster, tombstones=kept))

        return self._decode(self._document.update(build).record)

    # -- transaction machinery --------------------------------------------------------------------

    def _now(self) -> int:
        seconds = int(require_aware(self._clock(), "clock").timestamp())
        if seconds < 0:
            raise ValueError("clock must not be before 1970-01-01")
        return seconds

    def _require(self, current: StoredDocument | None) -> Roster:
        if current is None:
            raise SecureStoreError(StoreFault.IDENTITY_MISSING, f"{self.path} does not exist")
        return self._decode(current.record)

    def _apply(
        self,
        change: Callable[[Roster, int], tuple[Roster, RosterChange]],
        expected_roster_generation: int | None,
    ) -> RosterChange:
        """Run ``change`` on the verified current roster, with the next cursor value, in one locked update."""
        if expected_roster_generation is not None and not _is_uint64(expected_roster_generation):
            raise ValueError("expected_roster_generation must be an unsigned 64-bit integer")
        produced: list[RosterChange] = []

        def build(current: StoredDocument | None) -> Mapping[str, JsonValue]:
            roster = self._require(current)
            if expected_roster_generation is not None and expected_roster_generation != roster.generation:
                raise RosterRefused(RosterRefusal.STALE_GENERATION, f"the roster generation is {roster.generation}")
            if roster.generation >= _UINT64_MAX:
                raise RosterRefused(RosterRefusal.GENERATION_EXHAUSTED, "no further roster generation exists")
            changed, event = change(roster, roster.generation + 1)
            produced.clear()
            produced.append(event)
            return self._encode(changed)

        self._document.update(build)
        return produced[0]

    # -- encoding ---------------------------------------------------------------------------------

    @staticmethod
    def _encode(roster: Roster) -> Mapping[str, JsonValue]:
        reporters: list[JsonValue] = [
            {
                "enrolled_at": r.enrolled_at,
                "jwk": r.jwk.decode("ascii"),
                "key_generation": r.key_generation,
                "key_thumbprint": r.key_thumbprint,
                "metadata": r.metadata.to_json(),
                "reporter_id": r.reporter_id,
                "revoked_at": r.revoked_at,
                "state": r.state.value,
                "state_generation": r.state_generation,
            }
            for r in roster.reporters
        ]
        tombstones: list[JsonValue] = [
            {
                "deleted_at_generation": t.deleted_at_generation,
                "reporter_id": t.reporter_id,
                "state_generation": t.state_generation,
            }
            for t in roster.tombstones
        ]
        return {"generation": roster.generation, "reporters": reporters, "tombstones": tombstones}

    def _decode(self, record: Mapping[str, JsonValue]) -> Roster:
        malformed = SecureStoreError(StoreFault.MALFORMED, f"{self.path} is not a reporter roster record")
        if set(record) != _RECORD_KEYS:
            raise malformed
        generation, raw_reporters, raw_tombstones = record["generation"], record["reporters"], record["tombstones"]
        if (
            not _is_uint64(generation)
            or type(generation) is not int
            or type(raw_reporters) is not list
            or type(raw_tombstones) is not list
            or len(raw_reporters) > MAX_REPORTERS
            or len(raw_tombstones) > MAX_TOMBSTONES
        ):
            raise malformed
        try:
            reporters = tuple(self._decode_reporter(item) for item in raw_reporters)
            tombstones = tuple(self._decode_tombstone(item) for item in raw_tombstones)
        except (ProtocolViolation, ValueError):
            raise malformed from None
        self._check_invariants(generation, reporters, tombstones)
        return Roster(generation, reporters, tombstones)

    @staticmethod
    def _decode_reporter(raw: JsonValue) -> ReporterRecord:
        if type(raw) is not dict or set(raw) != _REPORTER_KEYS:
            raise ValueError("reporter")
        reporter_id, jwk, thumbprint, state = raw["reporter_id"], raw["jwk"], raw["key_thumbprint"], raw["state"]
        key_generation, state_generation = raw["key_generation"], raw["state_generation"]
        enrolled_at, revoked_at = raw["enrolled_at"], raw["revoked_at"]
        if (
            type(reporter_id) is not str
            or type(jwk) is not str
            or type(thumbprint) is not str
            or type(state) is not str
            or type(key_generation) is not int
            or type(state_generation) is not int
            or type(enrolled_at) is not int
            or not (revoked_at is None or type(revoked_at) is int)
            or not all(_is_uint64(v) for v in (key_generation, state_generation, enrolled_at))
            or not (revoked_at is None or _is_uint64(revoked_at))
        ):
            raise ValueError("reporter fields")
        _check_reporter_id(reporter_id)
        parsed = parse_p256_public_jwk(jwk.encode("ascii"))
        if parsed.thumbprint != thumbprint:
            raise ValueError("key thumbprint")
        return ReporterRecord(
            reporter_id=reporter_id,
            state=ReporterState(state),
            jwk=parsed.canonical_bytes,
            key_thumbprint=thumbprint,
            key_generation=key_generation,
            state_generation=state_generation,
            metadata=ReporterMetadata.from_json(raw["metadata"]),
            enrolled_at=enrolled_at,
            revoked_at=revoked_at,
        )

    @staticmethod
    def _decode_tombstone(raw: JsonValue) -> Tombstone:
        if type(raw) is not dict or set(raw) != _TOMBSTONE_KEYS:
            raise ValueError("tombstone")
        reporter_id, state_generation, deleted_at = (
            raw["reporter_id"],
            raw["state_generation"],
            raw["deleted_at_generation"],
        )
        if type(reporter_id) is not str or type(state_generation) is not int or type(deleted_at) is not int:
            raise ValueError("tombstone fields")
        if not (_is_uint64(state_generation) and _is_uint64(deleted_at)):
            raise ValueError("tombstone numbers")
        _check_reporter_id(reporter_id)
        return Tombstone(reporter_id, state_generation, deleted_at)

    def _check_invariants(
        self, generation: int, reporters: tuple[ReporterRecord, ...], tombstones: tuple[Tombstone, ...]
    ) -> None:
        problems: list[str] = []
        ids = [r.reporter_id for r in reporters]
        if ids != sorted(set(ids)):
            problems.append("reporters are not unique and sorted by ID")
        tombstone_ids = [t.reporter_id for t in tombstones]
        if tombstone_ids != sorted(set(tombstone_ids)):
            problems.append("tombstones are not unique and sorted by ID")
        if set(ids) & set(tombstone_ids):
            problems.append("a reporter is both enrolled and tombstoned")
        if len({r.key_thumbprint for r in reporters}) != len(reporters):
            problems.append("two reporters share a signing key")
        for record in reporters:
            if record.key_generation < 1 or record.state_generation < record.key_generation:
                problems.append("a reporter's generations are inconsistent")
            elif record.state_generation > generation:
                problems.append("a reporter's state generation is ahead of the roster generation")
            if record.key_generation == 1 and record.key_thumbprint != record.reporter_id:
                problems.append("a reporter that never rotated is not under its own initial key")
            if (record.state is ReporterState.REVOKED) != (record.revoked_at is not None):
                problems.append("a reporter's revocation time disagrees with its state")
        if any(t.deleted_at_generation > generation or t.state_generation < 1 for t in tombstones):
            problems.append("a tombstone is ahead of the roster generation")
        if any(r.key_thumbprint in set(ids) - {r.reporter_id} for r in reporters):
            problems.append("a reporter's signing key is another reporter's ID")
        if problems:
            raise SecureStoreError(StoreFault.IDENTITY_INCONSISTENT, f"{self.path}: {'; '.join(sorted(set(problems)))}")


def _require_active(roster: Roster, reporter_id: str) -> ReporterRecord:
    record = roster.get(reporter_id)
    if record is None:
        raise RosterRefused(RosterRefusal.NOT_FOUND, "no such reporter")
    if record.state is not ReporterState.ACTIVE:
        raise RosterRefused(RosterRefusal.NOT_ACTIVE, "that reporter is not active")
    return record


def _replace_record(roster: Roster, updated: ReporterRecord, generation: int) -> Roster:
    reporters = tuple(updated if r.reporter_id == updated.reporter_id else r for r in roster.reporters)
    return replace(roster, generation=generation, reporters=reporters)
