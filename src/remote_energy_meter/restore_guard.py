"""Restore checks: detect a one-sided restore, and hold normal service during a declared full-system restore.

Requirement 5.3 and design.md "PKI and durable state" separate two restore cases that need different defences.

**One-sided restore** (the add-on alone, or Home Assistant alone, is restored). The two endpoints hold control
generations and pins independently, so a rolled-back side shows up as a disagreement. :func:`compare_control_state`
compares what the add-on stores with what the integration presents and says which side is behind, whether a pin was
substituted, or whether the peer holds a revoked key; anything but a match disables control.

**Coordinated full-system restore** (both sides roll back together). The restored pair is internally consistent, so
nothing can detect it ("no external monotonic witness"). The *declared* restore workflow instead places the add-on in a
durable post-restore quarantine before normal service starts. :class:`QuarantineStore` is that record, and
:class:`Quarantine` is the gate every service path asks:

* restored control authority cannot administer reporters until a **fresh pairing** has replaced it;
* **reporter traffic** (challenges, sessions, reports) is refused for everyone until every restored identity has been
  durably deleted, MQTT v2 re-provisioned and acknowledged, and control re-paired, so a restored identity that was
  somehow left out of the declared list is blocked too;
* **MQTT v2** stays disabled until it has been re-provisioned with a fresh key and acknowledged;
* the quarantine is released only when all three are done, and the later steps need the fresh pairing first: the new
  control provisions MQTT and performs the deletions.

The record is created explicitly (:meth:`QuarantineStore.create`, by the add-on's first start, like the other durable
identities). A record that is absent, damaged or tampered is an error from the authenticated store, and
:func:`evaluate_gate` turns every such error into "blocked", so a missing or partial marker fails closed. Only
:meth:`QuarantineStore.declare_restore` may start from an absent record: it is the recovery path after a restore that
left none.

What this module cannot do, and the open decision: it records and enforces the quarantine, but nothing here *notices* a
restore. The add-on cannot tell on its own that Supervisor restored its data, and a restore overwrites ``/data``
including any marker, so the declaration must come from outside, after the data is restored and before normal service
starts (for example an explicit administrator action in the integration's recovery flow, passed to the add-on through
its protected options). How the declaration reaches the add-on belongs to task 7.4. Until it is wired, a restore
that is never declared is not quarantined.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from remote_energy_meter.authenticated_store import AuthenticatedDocumentStore, StateAuthenticationKeyStore
from remote_energy_meter.control_authority import ControlAuthority
from remote_energy_meter.key_material import require_aware
from remote_energy_meter.protocol_v1 import JsonValue, ProtocolViolation, decode_base64url
from remote_energy_meter.secure_store import SecureStoreError, StoredDocument, StoreFault, ensure_private_directory

__all__ = [
    "DOCUMENT_NAME",
    "MAX_RESTORED_REPORTERS",
    "ControlComparison",
    "GateDecision",
    "PeerControlState",
    "Quarantine",
    "QuarantineRefusal",
    "QuarantineRefused",
    "QuarantineStore",
    "compare_control_state",
    "evaluate_gate",
]

DOCUMENT_NAME: Final = "post-restore-quarantine.json"
MAX_RESTORED_REPORTERS: Final = 1000
"""The roster holds at most 1,000 enrolled reporters (requirement 8.1), so no more can be pending disposition."""

_UINT64_MAX: Final = 2**64 - 1
_STORE_ID: Final = "post-restore-quarantine"
_SCHEMA_VERSION: Final = 1
_MAX_BYTES: Final = 128 * 1024
_STATES: Final = frozenset({"clear", "quarantined"})
_RECORD_KEYS: Final = frozenset(
    {"control_repaired", "declared_at", "mqtt_reprovisioned", "pending_reporters", "restore_generation", "state"}
)


# --------------------------------------------------------------------------------------------------
# one-sided restore: compare the two endpoints' control state
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PeerControlState:
    """The control state the other endpoint presents: its key pin, SPKI pin, and the generations it holds."""

    key_thumbprint: str
    key_generation: int
    spki_sha256: str
    spki_generation: int
    recovery_generation: int


class ControlComparison(StrEnum):
    """How the add-on's stored control state relates to the peer's. Only ``MATCH`` leaves control enabled."""

    MATCH = "match"
    """Both sides hold the same key, SPKI, and generations."""

    NOT_PAIRED = "not_paired"
    """The add-on has no control authority and the peer claims nothing newer: there is nothing to administer."""

    LOCAL_BEHIND = "local_behind"
    """The peer holds newer generations than the add-on: the add-on's data was rolled back."""

    PEER_BEHIND = "peer_behind"
    """The add-on holds newer generations than the peer: the integration's data was rolled back."""

    PEER_REVOKED = "peer_revoked"
    """The peer holds a key or SPKI generation that the add-on has revoked."""

    SUBSTITUTED = "substituted"
    """Both sides name the same generations but different pins: one side's key or SPKI was replaced."""

    INCONSISTENT = "inconsistent"
    """The generations disagree in opposite directions, so neither side is simply behind."""

    @property
    def control_enabled(self) -> bool:
        """Whether control may proceed. Every outcome except a match disables it."""
        return self is ControlComparison.MATCH


def compare_control_state(local: ControlAuthority, peer: PeerControlState) -> ControlComparison:
    """Compare the add-on's stored control state with what the integration presents.

    Order of the decision, so the result is deterministic:

    1. A peer generation (1 or more; 0 means nothing held) at or below a locally revoked mark is ``PEER_REVOKED``.
    2. With no local authority, a peer claiming a generation the add-on never issued is ``LOCAL_BEHIND``; otherwise the
       add-on is ``NOT_PAIRED``.
    3. With a local authority, compare the key, SPKI, and recovery generations. All equal: the pins must match
       (``MATCH`` or ``SUBSTITUTED``). Differences all in one direction are ``LOCAL_BEHIND`` (the peer is ahead) or
       ``PEER_BEHIND``; differences in opposite directions are ``INCONSISTENT``.

    Raises:
        ValueError: A generation in ``peer`` is not an unsigned 64-bit integer.
    """
    for value in (peer.key_generation, peer.spki_generation, peer.recovery_generation):
        if type(value) is not int or not 0 <= value <= _UINT64_MAX:
            raise ValueError("peer generations must be unsigned 64-bit integers")
    # Generation 0 means "nothing held" (a peer that never paired), which is never a revoked generation.
    if (0 < peer.key_generation <= local.key_revoked_through) or (
        0 < peer.spki_generation <= local.spki_revoked_through
    ):
        return ControlComparison.PEER_REVOKED
    held = local.authority
    if held is None:
        issued_further = (
            peer.key_generation > local.key_high_water or peer.recovery_generation > local.recovery_high_water
        )
        return ControlComparison.LOCAL_BEHIND if issued_further else ControlComparison.NOT_PAIRED
    differences = (
        peer.key_generation - held.key_generation,
        peer.spki_generation - held.spki_generation,
        peer.recovery_generation - local.recovery_high_water,
    )
    if any(d > 0 for d in differences) and any(d < 0 for d in differences):
        return ControlComparison.INCONSISTENT
    if any(d > 0 for d in differences):
        return ControlComparison.LOCAL_BEHIND
    if any(d < 0 for d in differences):
        return ControlComparison.PEER_BEHIND
    if peer.key_thumbprint == held.thumbprint and peer.spki_sha256 == held.spki_sha256:
        return ControlComparison.MATCH
    return ControlComparison.SUBSTITUTED


# --------------------------------------------------------------------------------------------------
# coordinated restore: the durable quarantine and its gate
# --------------------------------------------------------------------------------------------------


class QuarantineRefusal(StrEnum):
    """Why a quarantine change was refused. A refusal changes nothing."""

    NOT_QUARANTINED = "not_quarantined"
    CONTROL_NOT_REPAIRED = "control_not_repaired"
    OBLIGATIONS_OUTSTANDING = "obligations_outstanding"


class QuarantineRefused(Exception):  # noqa: N818 - a refusal, not an error: the record is simply unchanged
    """A requested change violates a quarantine rule; nothing was written. ``reason`` is the rule."""

    def __init__(self, reason: QuarantineRefusal, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Quarantine:
    """The verified quarantine record, and the gate every service path asks."""

    quarantined: bool
    restore_generation: int
    declared_at: int
    control_repaired: bool
    mqtt_reprovisioned: bool
    pending_reporters: tuple[str, ...]

    @property
    def normal_service_permitted(self) -> bool:
        """Whether every service path is open, which is exactly when the add-on is not quarantined."""
        return not self.quarantined

    @property
    def control_administration_permitted(self) -> bool:
        """Whether the control authority may administer reporters.

        The restored authority may not; the fresh one that replaced it after re-pairing may, because it is what deletes
        the restored identities.
        """
        return not self.quarantined or self.control_repaired

    @property
    def mqtt_v2_permitted(self) -> bool:
        """Whether MQTT v2 may connect, publish, or apply retained state. Never while quarantined."""
        return not self.quarantined

    @property
    def reporter_traffic_permitted(self) -> bool:
        """Whether any reporter may obtain a challenge or a session, or have a report accepted.

        While quarantined nobody may: the gates stay closed until every obligation is done, so a restored identity is
        blocked even if it was missing from the declared list. Enrollment from scratch is a separate path and is not
        gated here.
        """
        return not self.quarantined

    @property
    def outstanding(self) -> tuple[str, ...]:
        """The obligations still open, as stable names: ``control_repair``, ``mqtt_reprovision``, ``reporter:<id>``."""
        if not self.quarantined:
            return ()
        items = [] if self.control_repaired else ["control_repair"]
        if not self.mqtt_reprovisioned:
            items.append("mqtt_reprovision")
        return (*items, *(f"reporter:{reporter}" for reporter in self.pending_reporters))


_CLEAR: Final = Quarantine(False, 0, 0, False, False, ())


@dataclass(frozen=True, slots=True)
class GateDecision:
    """The fail-closed answer to "may normal service run?": a verdict and, when blocked, a stable reason."""

    permitted: bool
    reason: str | None = None
    quarantine: Quarantine | None = None


def evaluate_gate(store: QuarantineStore) -> GateDecision:
    """Decide whether normal service may run, failing closed on any damage to the record.

    A record that is tampered, rolled back, insecure, or unreadable raises from the authenticated store; every such
    error blocks service here instead of being treated as "no quarantine".
    """
    try:
        state = store.load()
    except SecureStoreError as error:
        return GateDecision(False, f"quarantine_record_{error.fault.value}")
    if state.quarantined:
        return GateDecision(False, "post_restore_quarantine", state)
    return GateDecision(True, None, state)


class QuarantineStore:
    """Durable, authenticated storage of the post-restore quarantine.

    Args:
        directory: A dedicated owner-only directory for this record.
        keys: The state-authentication key store whose key tags the record.
        clock: Returns the current time as a timezone-aware ``datetime``; stored as the declaration time.
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
        """The record's path."""
        return self._document.path

    def create(self) -> Quarantine:
        """Write the initial clear record, once. Call it when the add-on first initializes its durable state.

        Raises:
            SecureStoreError: ``IDENTITY_EXISTS`` when a record already exists (it is never replaced), including one
                created while this call runs; any other fault of the authenticated store.
        """
        ensure_private_directory(self._directory)
        if self._document.load() is not None:
            raise SecureStoreError(StoreFault.IDENTITY_EXISTS, f"{self.path} already exists and is never replaced")
        try:
            self._document.create(self._encode(_CLEAR))
        except SecureStoreError as error:
            if error.fault is StoreFault.CONCURRENT_MODIFICATION:
                detail = f"{self.path} was created by another writer and is never replaced"
                raise SecureStoreError(StoreFault.IDENTITY_EXISTS, detail) from None
            raise
        return _CLEAR

    def load(self) -> Quarantine:
        """Read and verify the record.

        Raises:
            SecureStoreError: ``IDENTITY_MISSING`` when no record exists (a missing marker fails closed); any fault of
                the authenticated store; ``MALFORMED`` or ``IDENTITY_INCONSISTENT`` for a record that breaks its rules.
        """
        document = self._document.load()
        if document is None:
            raise SecureStoreError(StoreFault.IDENTITY_MISSING, f"{self.path} does not exist")
        return self._decode(document.record)

    def declare_restore(self, restored_reporters: Sequence[str]) -> Quarantine:
        """Place the add-on in quarantine for a declared restore, resetting every obligation.

        Declaring again, for example after restoring a backup that was itself taken during a quarantine, starts
        over: the flags are cleared and the pending set is replaced, so stale progress carried in restored data can
        never count. This is the only change that may start from an absent record, because a restore may have left none.

        Args:
            restored_reporters: The identities that came back with the restored roster, each a canonical 43-character
                base64url reporter ID; duplicates are collapsed.

        Raises:
            ValueError: An identifier is malformed or more than :data:`MAX_RESTORED_REPORTERS` are given.
            SecureStoreError: Any fault of the authenticated store.
        """
        reporters = _canonical_reporters(restored_reporters)
        declared_at = int(require_aware(self._clock(), "clock").timestamp())

        def change(current: Quarantine) -> Quarantine:
            return Quarantine(True, current.restore_generation + 1, declared_at, False, False, reporters)

        return self._update(change, allow_absent=True)

    def mark_control_repaired(self) -> Quarantine:
        """Record that a fresh control pairing has replaced the restored authority.

        Raises:
            QuarantineRefused: ``NOT_QUARANTINED``.
        """
        return self._update(lambda current: replace(_require_quarantined(current), control_repaired=True))

    def mark_mqtt_reprovisioned(self) -> Quarantine:
        """Record that MQTT v2 was re-provisioned with a fresh key and the new record was acknowledged.

        The fresh control provisions it, so the re-pairing must come first.

        Raises:
            QuarantineRefused: ``NOT_QUARANTINED`` or ``CONTROL_NOT_REPAIRED``.
        """
        return self._update(lambda current: replace(_require_repaired(current), mqtt_reprovisioned=True))

    def dispose_reporter(self, reporter_id: str) -> Quarantine:
        """Record that a restored reporter identity was durably deleted. Idempotent: a retry after a crash is a no-op.

        Only the fresh control authority can delete an identity, so the re-pairing must come first.

        Raises:
            ValueError: ``reporter_id`` is malformed.
            QuarantineRefused: ``NOT_QUARANTINED`` or ``CONTROL_NOT_REPAIRED``.
        """
        _check_reporter(reporter_id)

        def change(current: Quarantine) -> Quarantine:
            held = _require_repaired(current)
            return replace(held, pending_reporters=tuple(r for r in held.pending_reporters if r != reporter_id))

        return self._update(change)

    def release(self) -> Quarantine:
        """Leave quarantine, only when every obligation is done.

        Raises:
            QuarantineRefused: ``NOT_QUARANTINED``, or ``OBLIGATIONS_OUTSTANDING`` naming what is still open.
        """

        def change(current: Quarantine) -> Quarantine:
            held = _require_quarantined(current)
            if held.outstanding:
                raise QuarantineRefused(
                    QuarantineRefusal.OBLIGATIONS_OUTSTANDING, f"still outstanding: {', '.join(held.outstanding)}"
                )
            return Quarantine(False, held.restore_generation, held.declared_at, False, False, ())

        return self._update(change)

    # -- internals --------------------------------------------------------------------------------

    def _update(self, change: Callable[[Quarantine], Quarantine], *, allow_absent: bool = False) -> Quarantine:
        def build(current: StoredDocument | None) -> Mapping[str, JsonValue]:
            if current is None and not allow_absent:
                raise SecureStoreError(StoreFault.IDENTITY_MISSING, f"{self.path} does not exist")
            state = _CLEAR if current is None else self._decode(current.record)
            return self._encode(change(state))

        return self._decode(self._document.update(build).record)

    @staticmethod
    def _encode(state: Quarantine) -> Mapping[str, JsonValue]:
        return {
            "control_repaired": state.control_repaired,
            "declared_at": state.declared_at,
            "mqtt_reprovisioned": state.mqtt_reprovisioned,
            "pending_reporters": list(state.pending_reporters),
            "restore_generation": state.restore_generation,
            "state": "quarantined" if state.quarantined else "clear",
        }

    def _decode(self, record: Mapping[str, JsonValue]) -> Quarantine:
        malformed = SecureStoreError(StoreFault.MALFORMED, f"{self.path} is not a quarantine record")
        if set(record) != _RECORD_KEYS:
            raise malformed
        state, repaired, reprovisioned = record["state"], record["control_repaired"], record["mqtt_reprovisioned"]
        declared_at, generation, pending = (
            record["declared_at"],
            record["restore_generation"],
            record["pending_reporters"],
        )
        if (
            type(state) is not str
            or state not in _STATES
            or type(repaired) is not bool
            or type(reprovisioned) is not bool
            or type(declared_at) is not int
            or type(generation) is not int
            or not 0 <= declared_at <= _UINT64_MAX
            or not 0 <= generation <= _UINT64_MAX
            or type(pending) is not list
            or not all(type(item) is str for item in pending)
        ):
            raise malformed
        try:
            reporters = _canonical_reporters([str(item) for item in pending])  # each already checked to be a str
        except ValueError:
            raise malformed from None
        if reporters != tuple(pending):
            raise malformed
        quarantined = state == "quarantined"
        if not quarantined and (repaired or reprovisioned or reporters):
            raise SecureStoreError(StoreFault.IDENTITY_INCONSISTENT, f"{self.path}: a clear record carries obligations")
        if quarantined and generation < 1:
            raise SecureStoreError(
                StoreFault.IDENTITY_INCONSISTENT, f"{self.path}: a quarantine has no restore generation"
            )
        return Quarantine(quarantined, generation, declared_at, repaired, reprovisioned, reporters)


def _require_quarantined(current: Quarantine) -> Quarantine:
    if not current.quarantined:
        raise QuarantineRefused(QuarantineRefusal.NOT_QUARANTINED, "the add-on is not in post-restore quarantine")
    return current


def _require_repaired(current: Quarantine) -> Quarantine:
    held = _require_quarantined(current)
    if not held.control_repaired:
        raise QuarantineRefused(
            QuarantineRefusal.CONTROL_NOT_REPAIRED,
            "the restored control authority has not been replaced by a fresh pairing",
        )
    return held


def _check_reporter(reporter_id: str) -> None:
    try:
        decode_base64url(reporter_id, exact_length=32)
    except ProtocolViolation:
        raise ValueError("a reporter ID is a canonical 43-character base64url RFC 7638 thumbprint") from None


def _canonical_reporters(reporters: Sequence[str]) -> tuple[str, ...]:
    for reporter in reporters:
        _check_reporter(reporter)
    unique = sorted(set(reporters))
    if len(unique) > MAX_RESTORED_REPORTERS:
        raise ValueError(f"at most {MAX_RESTORED_REPORTERS} reporters can be pending")
    return tuple(unique)
