"""The durable control authority: who may administer reporters, and what was consumed to get there.

One authenticated document holds everything that must change together when control authority is created or revoked
(requirements 5.3, 7.5a, 7.5e and design.md "PKI and durable state"), so a crash leaves either the old state or the
new one, never a mixture:

* the **authorized integration control key** (its canonical public JWK and thumbprint), the **key generation** it was
  authorized under, and the **control-SPKI pin and generation** it was authorized against;
* the **revocation marks**: generations only increase, so revoking the current generation ``g`` makes every
  generation up to ``g`` dead (``key_revoked_through`` / ``spki_revoked_through``), plus the **recovery high-water
  mark**;
* the **consumed credential digests** (SHA-256 of the full pairing credential) with their absolute ``expires_at``, kept
  through ``expires_at + 300`` seconds;
* the **exact results** of the two consuming transactions, so a response lost in flight can be returned byte for byte:
  the completed bootstrap confirm and the completed recovery revoke. A result is returnable while ``now < expires_at``;
  from that instant only its non-returning tombstone remains, through ``expires_at + 300`` seconds.

Two operations change state, each in one locked read-verify-write of the authenticated document:

* :meth:`ControlAuthorityStore.authorize_pairing` authorizes the integration key, consumes the credential, and records
  the confirm result together;
* :meth:`ControlAuthorityStore.revoke_for_recovery` consumes a fresh credential, advances the recovery high-water mark,
  revokes the named generations, clears the authority, and records the revoke result together.

Every instant is an absolute Unix second embedded in a record or supplied by the caller from server receipt time.
Nothing derives an expiry from the current time, so restoring an older backup cannot restart or extend a lifetime: an
expired credential is refused on its embedded expiry even when its consumed digest is absent, and an expired result is
never returned. Physical compaction (dropping old digests and result bodies) runs on each write but no lookup depends
on it.

This module is state and invariants only. It does not verify HMAC proofs, parse transcripts, generate credentials,
serve routes, or invalidate leases and request nonces (those belong to task 7.4 and to the replay and result stores).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from remote_energy_meter.authenticated_store import AuthenticatedDocumentStore, StateAuthenticationKeyStore
from remote_energy_meter.protocol_v1 import (
    JsonValue,
    ProtocolViolation,
    decode_base64url,
    encode_base64url,
    parse_p256_public_jwk,
)
from remote_energy_meter.protocol_v1.encoding import decode_opaque_base64url
from remote_energy_meter.secure_store import SecureStoreError, StoredDocument, StoreFault

__all__ = [
    "DOCUMENT_NAME",
    "MAX_CONSUMED",
    "MAX_RESPONSE_BYTES",
    "MAX_RESULTS",
    "TOMBSTONE_GRACE_SECONDS",
    "AuthorityRefusal",
    "AuthorityRefused",
    "AuthorizedControl",
    "ConsumedCredential",
    "ControlAuthority",
    "ControlAuthorityStore",
    "ExactResult",
    "ResultKind",
    "ResultLookup",
    "RevocationScope",
]

DOCUMENT_NAME: Final = "control-authority.json"
TOMBSTONE_GRACE_SECONDS: Final = 300
"""A consumed-credential digest and a result tombstone are kept this long after the credential's embedded expiry."""
MAX_CONSUMED: Final = 64
MAX_RESULTS: Final = 32
MAX_RESPONSE_BYTES: Final = 2048
"""Cap on a recorded response body; the bootstrap and recovery responses are well under 1 KiB."""

_UINT64_MAX: Final = 2**64 - 1
_STORE_ID: Final = "control-authority"
_SCHEMA_VERSION: Final = 1
_MAX_BYTES: Final = 256 * 1024
_RECORD_KEYS: Final = frozenset(
    {
        "authority",
        "consumed",
        "key_high_water",
        "key_revoked_through",
        "recovery_high_water",
        "results",
        "spki_revoked_through",
    }
)
_AUTHORITY_KEYS: Final = frozenset(
    {"authorized_at", "jwk", "key_generation", "spki_generation", "spki_sha256", "thumbprint"}
)
_CONSUMED_KEYS: Final = frozenset({"digest", "expires_at"})
_RESULT_KEYS: Final = frozenset({"expires_at", "id", "kind", "request_hash", "response"})


class ResultKind(StrEnum):
    """Which consuming transaction a recorded result belongs to."""

    BOOTSTRAP_CONFIRM = "bootstrap-confirm"
    RECOVERY_REVOKE = "recovery-revoke"


class RevocationScope(StrEnum):
    """What a recovery revokes. Each value is the literal wire value of requirement 7.5e."""

    CONTROL_KEY = "control-key"
    CONTROL_SPKI = "control-spki"
    CONTROL_KEY_AND_SPKI = "control-key-and-spki"


class AuthorityRefusal(StrEnum):
    """Why a state change was refused. A refusal changes nothing."""

    ALREADY_PAIRED = "already_paired"
    NOT_PAIRED = "not_paired"
    CREDENTIAL_CONSUMED = "credential_consumed"
    CREDENTIAL_EXPIRED = "credential_expired"
    KEY_GENERATION_MISMATCH = "key_generation_mismatch"
    RECOVERY_GENERATION_MISMATCH = "recovery_generation_mismatch"
    SPKI_REVOKED = "spki_revoked"
    CURRENT_STATE_MISMATCH = "current_state_mismatch"
    RESULT_ID_REUSED = "result_id_reused"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    GENERATION_EXHAUSTED = "generation_exhausted"


class AuthorityRefused(Exception):  # noqa: N818 - a refusal, not an error: the state is simply unchanged
    """A requested change violates an authority rule; nothing was written. ``reason`` is the rule."""

    def __init__(self, reason: AuthorityRefusal, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class AuthorizedControl:
    """The currently authorized integration control key and the SPKI it was authorized against."""

    jwk: bytes
    thumbprint: str
    key_generation: int
    spki_sha256: str
    spki_generation: int
    authorized_at: int


@dataclass(frozen=True, slots=True)
class ConsumedCredential:
    """The digest of a consumed pairing credential and the absolute instant it expires."""

    digest: bytes
    expires_at: int


@dataclass(frozen=True, slots=True)
class ExactResult:
    """A completed consuming transaction's request hash and exact response, keyed by its random identifier."""

    kind: ResultKind
    identifier: str
    request_hash: bytes
    response: bytes | None
    expires_at: int


@dataclass(frozen=True, slots=True)
class ResultLookup:
    """The outcome of an exact-retry lookup.

    ``response`` is set only for an exact match inside the return window. ``conflict`` is true when the identifier is
    known inside the window but the request hash differs, which the caller reports as ``409``.
    """

    response: bytes | None = None
    conflict: bool = False


@dataclass(frozen=True, slots=True)
class ControlAuthority:
    """A verified snapshot of the control authority."""

    authority: AuthorizedControl | None
    key_high_water: int
    key_revoked_through: int
    spki_revoked_through: int
    recovery_high_water: int
    consumed: tuple[ConsumedCredential, ...]
    results: tuple[ExactResult, ...]

    @property
    def next_key_generation(self) -> int:
        """The generation the next pairing is authorized under: one for the first pairing, never a reused number."""
        return self.key_high_water + 1

    @property
    def next_recovery_generation(self) -> int:
        """The recovery generation the next recovery must name."""
        return self.recovery_high_water + 1

    def is_consumed(self, digest: bytes) -> bool:
        """Whether ``digest`` is a recorded consumed credential (its tombstone may already have been compacted)."""
        return any(entry.digest == digest for entry in self.consumed)


_EMPTY: Final = ControlAuthority(None, 0, 0, 0, 0, (), ())


def _check_digest(value: bytes, name: str) -> None:
    if type(value) is not bytes or len(value) != 32:
        raise ValueError(f"{name} must be exactly 32 bytes")


def _check_instant(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value <= _UINT64_MAX:
        raise ValueError(f"{name} must be a uint64 Unix time")


def _check_identifier(value: str, name: str) -> None:
    try:
        decode_base64url(value, exact_length=16)
    except ProtocolViolation:
        raise ValueError(f"{name} must be a canonical 22-character base64url value") from None


class ControlAuthorityStore:
    """Durable, authenticated storage of the control authority and its consuming transactions.

    Args:
        directory: A dedicated owner-only directory for this document.
        keys: The state-authentication key store whose key tags the record.
        clock: Returns the current time as a timezone-aware ``datetime``; it is read as server receipt time.
    """

    def __init__(
        self,
        directory: Path,
        *,
        keys: StateAuthenticationKeyStore,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._clock = clock
        self._lock = threading.Lock()
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

    def load(self) -> ControlAuthority:
        """Read and verify the authority; a never-written store is the empty authority (nothing paired).

        Raises:
            SecureStoreError: Any fault of the authenticated store, or ``MALFORMED``/``IDENTITY_INCONSISTENT`` for a
                record that is authentic but violates the invariants of this module.
        """
        document = self._document.load()
        return _EMPTY if document is None else self._decode(document.record)

    def lookup_result(self, kind: ResultKind, identifier: str, request_hash: bytes) -> ResultLookup:
        """Look up an exact retry. Reads only; the return window is decided by the recorded expiry, never by cleanup.

        Raises:
            ValueError: ``identifier`` or ``request_hash`` is malformed.
        """
        _check_identifier(identifier, "identifier")
        _check_digest(request_hash, "request_hash")
        now = self._now()
        for result in self.load().results:
            if result.kind is kind and result.identifier == identifier:
                if now >= result.expires_at or result.response is None:
                    return ResultLookup()
                if result.request_hash != request_hash:
                    return ResultLookup(conflict=True)
                return ResultLookup(response=result.response)
        return ResultLookup()

    def authorize_pairing(
        self,
        *,
        jwk: bytes,
        spki_sha256: str,
        spki_generation: int,
        key_generation: int,
        credential_digest: bytes,
        credential_expires_at: int,
        bootstrap_id: str,
        request_hash: bytes,
        response: bytes,
        result_expires_at: int,
    ) -> ControlAuthority:
        """Authorize the integration key, consume the credential, and record the confirm result, atomically.

        Args:
            jwk: The integration's exact canonical public JWK bytes.
            spki_sha256: The control-SPKI pin the pairing was bound to.
            spki_generation: That SPKI's generation.
            key_generation: The generation this key is authorized under; must equal ``next_key_generation``, so the
                response the caller built (which names it) cannot disagree with what is stored.
            credential_digest: SHA-256 of the complete canonical pairing credential.
            credential_expires_at: The credential's embedded absolute expiry (Unix seconds).
            bootstrap_id: The completed bootstrap's identifier (canonical 22-character base64url).
            request_hash: SHA-256 of the exact confirm request.
            response: The exact success response bytes to return on a retry.
            result_expires_at: The last instant, exclusive, at which the response may be returned.

        Raises:
            ValueError: An argument is malformed.
            AuthorityRefused: ``ALREADY_PAIRED``, ``CREDENTIAL_CONSUMED``, ``CREDENTIAL_EXPIRED``,
                ``KEY_GENERATION_MISMATCH``, ``SPKI_REVOKED``, ``RESULT_ID_REUSED``, ``CAPACITY_EXCEEDED`` or
                ``GENERATION_EXHAUSTED``.
            SecureStoreError: Any fault of the authenticated store.
        """
        parsed = _parse_jwk(jwk)
        _check_digest(credential_digest, "credential_digest")
        _check_digest(request_hash, "request_hash")
        _check_identifier(bootstrap_id, "bootstrap_id")
        _check_digest_text(spki_sha256, "spki_sha256")
        for value, name in (
            (spki_generation, "spki_generation"),
            (key_generation, "key_generation"),
            (credential_expires_at, "credential_expires_at"),
            (result_expires_at, "result_expires_at"),
        ):
            _check_instant(value, name)
        _check_response(response)
        thumbprint = parsed.thumbprint

        def change(state: ControlAuthority, now: int) -> ControlAuthority:
            if state.authority is not None:
                raise AuthorityRefused(AuthorityRefusal.ALREADY_PAIRED, "control authority is already paired")
            self._require_fresh_credential(state, credential_digest, credential_expires_at, now)
            if state.key_high_water >= _UINT64_MAX:
                raise AuthorityRefused(AuthorityRefusal.GENERATION_EXHAUSTED, "no further key generation exists")
            if key_generation != state.next_key_generation:
                raise AuthorityRefused(
                    AuthorityRefusal.KEY_GENERATION_MISMATCH, f"the next key generation is {state.next_key_generation}"
                )
            if spki_generation <= state.spki_revoked_through:
                raise AuthorityRefused(AuthorityRefusal.SPKI_REVOKED, "that control SPKI generation is revoked")
            result = ExactResult(ResultKind.BOOTSTRAP_CONFIRM, bootstrap_id, request_hash, response, result_expires_at)
            self._require_new_result(state, result)
            authority = AuthorizedControl(
                jwk=parsed.canonical_bytes,
                thumbprint=thumbprint,
                key_generation=key_generation,
                spki_sha256=spki_sha256,
                spki_generation=spki_generation,
                authorized_at=now,
            )
            changed = replace(
                state,
                authority=authority,
                key_high_water=key_generation,
                consumed=(*state.consumed, ConsumedCredential(credential_digest, credential_expires_at)),
                results=(*state.results, result),
            )
            return self._compact(changed, now)

        return self._apply(change)

    def revoke_for_recovery(
        self,
        *,
        scope: RevocationScope,
        key_thumbprint: str,
        key_generation: int,
        spki_sha256: str,
        spki_generation: int,
        recovery_generation: int,
        credential_digest: bytes,
        credential_expires_at: int,
        recovery_id: str,
        request_hash: bytes,
        response: bytes,
        result_expires_at: int,
    ) -> ControlAuthority:
        """Consume a fresh credential, revoke the named generations, and record the revoke result, atomically.

        The authority is cleared whatever the scope: a revoked key or SPKI is never reactivated, and re-pairing needs a
        second fresh credential. ``scope`` decides which generations become permanently unusable.

        Raises:
            ValueError: An argument is malformed.
            AuthorityRefused: ``NOT_PAIRED``, ``CURRENT_STATE_MISMATCH`` (the supplied current thumbprint, generation or
                SPKI is not the stored one), ``RECOVERY_GENERATION_MISMATCH``, ``CREDENTIAL_CONSUMED``,
                ``CREDENTIAL_EXPIRED``, ``RESULT_ID_REUSED`` or ``CAPACITY_EXCEEDED``.
            SecureStoreError: Any fault of the authenticated store.
        """
        _check_digest_text(key_thumbprint, "key_thumbprint")
        _check_digest_text(spki_sha256, "spki_sha256")
        _check_digest(credential_digest, "credential_digest")
        _check_digest(request_hash, "request_hash")
        _check_identifier(recovery_id, "recovery_id")
        for value, name in (
            (key_generation, "key_generation"),
            (spki_generation, "spki_generation"),
            (recovery_generation, "recovery_generation"),
            (credential_expires_at, "credential_expires_at"),
            (result_expires_at, "result_expires_at"),
        ):
            _check_instant(value, name)
        _check_response(response)

        def change(state: ControlAuthority, now: int) -> ControlAuthority:
            current = state.authority
            if current is None:
                raise AuthorityRefused(AuthorityRefusal.NOT_PAIRED, "there is no control authority to revoke")
            if (current.thumbprint, current.key_generation, current.spki_sha256, current.spki_generation) != (
                key_thumbprint,
                key_generation,
                spki_sha256,
                spki_generation,
            ):
                raise AuthorityRefused(AuthorityRefusal.CURRENT_STATE_MISMATCH, "the named current state is not stored")
            if recovery_generation != state.next_recovery_generation or state.recovery_high_water >= _UINT64_MAX:
                raise AuthorityRefused(
                    AuthorityRefusal.RECOVERY_GENERATION_MISMATCH,
                    f"the next recovery generation is {state.next_recovery_generation}",
                )
            self._require_fresh_credential(state, credential_digest, credential_expires_at, now)
            result = ExactResult(ResultKind.RECOVERY_REVOKE, recovery_id, request_hash, response, result_expires_at)
            self._require_new_result(state, result)
            revokes_key = scope is not RevocationScope.CONTROL_SPKI
            revokes_spki = scope is not RevocationScope.CONTROL_KEY
            changed = replace(
                state,
                authority=None,
                key_revoked_through=current.key_generation if revokes_key else state.key_revoked_through,
                spki_revoked_through=current.spki_generation if revokes_spki else state.spki_revoked_through,
                recovery_high_water=recovery_generation,
                consumed=(*state.consumed, ConsumedCredential(credential_digest, credential_expires_at)),
                results=(*state.results, result),
            )
            return self._compact(changed, now)

        return self._apply(change)

    # -- rules ------------------------------------------------------------------------------------

    @staticmethod
    def _require_fresh_credential(state: ControlAuthority, digest: bytes, expires_at: int, now: int) -> None:
        # The embedded expiry decides, so a credential whose consumed digest was compacted or lost is still refused.
        if now >= expires_at:
            raise AuthorityRefused(AuthorityRefusal.CREDENTIAL_EXPIRED, "the credential has expired")
        if state.is_consumed(digest):
            raise AuthorityRefused(AuthorityRefusal.CREDENTIAL_CONSUMED, "the credential was already consumed")
        if len(state.consumed) >= MAX_CONSUMED:
            raise AuthorityRefused(AuthorityRefusal.CAPACITY_EXCEEDED, "too many unexpired consumed credentials")

    @staticmethod
    def _require_new_result(state: ControlAuthority, result: ExactResult) -> None:
        if any(r.kind is result.kind and r.identifier == result.identifier for r in state.results):
            raise AuthorityRefused(AuthorityRefusal.RESULT_ID_REUSED, "that identifier already has a recorded result")
        if len(state.results) >= MAX_RESULTS:
            raise AuthorityRefused(AuthorityRefusal.CAPACITY_EXCEEDED, "too many unexpired recorded results")

    # -- transaction machinery --------------------------------------------------------------------

    def _now(self) -> int:
        moment = self._clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return int(moment.timestamp())

    def _apply(self, change: Callable[[ControlAuthority, int], ControlAuthority]) -> ControlAuthority:
        """Run ``change`` on the verified current state and write its result in one locked update."""
        with self._lock:
            now = self._now()

            def build(current: StoredDocument | None) -> Mapping[str, JsonValue]:
                state = _EMPTY if current is None else self._decode(current.record)
                return self._encode(change(state, now))

            return self._decode(self._document.update(build).record)

    @staticmethod
    def _compact(state: ControlAuthority, now: int) -> ControlAuthority:
        """Drop what ``now`` has made unnecessary: old consumed digests, expired result bodies and old tombstones."""
        consumed = tuple(c for c in state.consumed if now < c.expires_at + TOMBSTONE_GRACE_SECONDS)
        results = tuple(
            replace(r, response=r.response if now < r.expires_at else None)
            for r in state.results
            if now < r.expires_at + TOMBSTONE_GRACE_SECONDS
        )
        return replace(state, consumed=consumed, results=results)

    # -- encoding ---------------------------------------------------------------------------------

    def _encode(self, state: ControlAuthority) -> Mapping[str, JsonValue]:
        authority: JsonValue = None
        if state.authority is not None:
            held = state.authority
            authority = {
                "authorized_at": held.authorized_at,
                "jwk": held.jwk.decode("ascii"),
                "key_generation": held.key_generation,
                "spki_generation": held.spki_generation,
                "spki_sha256": held.spki_sha256,
                "thumbprint": held.thumbprint,
            }
        consumed: list[JsonValue] = [
            {"digest": encode_base64url(c.digest, exact_length=32), "expires_at": c.expires_at} for c in state.consumed
        ]
        results: list[JsonValue] = [
            {
                "expires_at": r.expires_at,
                "id": r.identifier,
                "kind": r.kind.value,
                "request_hash": encode_base64url(r.request_hash, exact_length=32),
                "response": None if r.response is None else encode_base64url(r.response, exact_length=len(r.response)),
            }
            for r in state.results
        ]
        return {
            "authority": authority,
            "consumed": consumed,
            "key_high_water": state.key_high_water,
            "key_revoked_through": state.key_revoked_through,
            "recovery_high_water": state.recovery_high_water,
            "results": results,
            "spki_revoked_through": state.spki_revoked_through,
        }

    def _decode(self, record: Mapping[str, JsonValue]) -> ControlAuthority:
        where = self.path
        malformed = SecureStoreError(StoreFault.MALFORMED, f"{where} is not a control authority record")
        if set(record) != _RECORD_KEYS:
            raise malformed
        raw_consumed, raw_results = record["consumed"], record["results"]
        key_high_water, key_revoked_through = record["key_high_water"], record["key_revoked_through"]
        recovery_high_water, spki_revoked_through = record["recovery_high_water"], record["spki_revoked_through"]
        if (
            type(key_high_water) is not int
            or type(key_revoked_through) is not int
            or type(recovery_high_water) is not int
            or type(spki_revoked_through) is not int
            or not all(
                0 <= v <= _UINT64_MAX
                for v in (key_high_water, key_revoked_through, recovery_high_water, spki_revoked_through)
            )
            or type(raw_consumed) is not list
            or type(raw_results) is not list
            or len(raw_consumed) > MAX_CONSUMED
            or len(raw_results) > MAX_RESULTS
        ):
            raise malformed
        try:
            authority = self._decode_authority(record["authority"])
            consumed = tuple(self._decode_consumed(item) for item in raw_consumed)
            results = tuple(self._decode_result(item) for item in raw_results)
        except (ProtocolViolation, ValueError):
            raise malformed from None
        problems = []
        if key_revoked_through > key_high_water:
            problems.append("a key generation is revoked that was never issued")
        if len({c.digest for c in consumed}) != len(consumed):
            problems.append("a credential digest is recorded twice")
        if len({(r.kind, r.identifier) for r in results}) != len(results):
            problems.append("a result identifier is recorded twice")
        if authority is not None and not (
            key_revoked_through < authority.key_generation <= key_high_water
            and authority.spki_generation > spki_revoked_through
        ):
            problems.append("the authorized generations are revoked or were never issued")
        if problems:
            raise SecureStoreError(StoreFault.IDENTITY_INCONSISTENT, f"{where}: {'; '.join(problems)}")
        return ControlAuthority(
            authority, key_high_water, key_revoked_through, spki_revoked_through, recovery_high_water, consumed, results
        )

    @staticmethod
    def _decode_authority(raw: JsonValue) -> AuthorizedControl | None:
        if raw is None:
            return None
        if type(raw) is not dict or set(raw) != _AUTHORITY_KEYS:
            raise ValueError("authority")
        jwk, thumbprint, spki = raw["jwk"], raw["thumbprint"], raw["spki_sha256"]
        key_generation, spki_generation = raw["key_generation"], raw["spki_generation"]
        authorized_at = raw["authorized_at"]
        if (
            type(jwk) is not str
            or type(thumbprint) is not str
            or type(spki) is not str
            or type(key_generation) is not int
            or type(spki_generation) is not int
            or type(authorized_at) is not int
            or not all(0 <= n <= _UINT64_MAX for n in (key_generation, spki_generation, authorized_at))
            or key_generation < 1
        ):
            raise ValueError("authority fields")
        parsed = _parse_jwk(jwk.encode("ascii"))
        _check_digest_text(spki, "spki_sha256")
        if parsed.thumbprint != thumbprint:
            raise ValueError("thumbprint")
        return AuthorizedControl(
            parsed.canonical_bytes, thumbprint, key_generation, spki, spki_generation, authorized_at
        )

    @staticmethod
    def _decode_consumed(raw: JsonValue) -> ConsumedCredential:
        if type(raw) is not dict or set(raw) != _CONSUMED_KEYS:
            raise ValueError("consumed")
        digest, expires_at = raw["digest"], raw["expires_at"]
        if type(digest) is not str or type(expires_at) is not int or not 0 <= expires_at <= _UINT64_MAX:
            raise ValueError("consumed fields")
        return ConsumedCredential(decode_base64url(digest, exact_length=32), expires_at)

    @staticmethod
    def _decode_result(raw: JsonValue) -> ExactResult:
        if type(raw) is not dict or set(raw) != _RESULT_KEYS:
            raise ValueError("result")
        kind, identifier, request_hash = raw["kind"], raw["id"], raw["request_hash"]
        response, expires_at = raw["response"], raw["expires_at"]
        if (
            type(kind) is not str
            or type(identifier) is not str
            or type(request_hash) is not str
            or not (response is None or type(response) is str)
            or type(expires_at) is not int
            or not 0 <= expires_at <= _UINT64_MAX
        ):
            raise ValueError("result fields")
        _check_identifier(identifier, "id")
        body = None if response is None else decode_opaque_base64url(response, minimum=1, maximum=MAX_RESPONSE_BYTES)
        return ExactResult(
            ResultKind(kind), identifier, decode_base64url(request_hash, exact_length=32), body, expires_at
        )


def _parse_jwk(raw: bytes) -> _Jwk:
    try:
        parsed = parse_p256_public_jwk(raw)
    except ProtocolViolation:
        raise ValueError("jwk must be an exact canonical public P-256 JWK") from None
    return _Jwk(parsed.canonical_bytes, parsed.thumbprint)


@dataclass(frozen=True, slots=True)
class _Jwk:
    canonical_bytes: bytes
    thumbprint: str


def _check_digest_text(value: str, name: str) -> None:
    try:
        decode_base64url(value, exact_length=32)
    except ProtocolViolation:
        raise ValueError(f"{name} must be a canonical 43-character base64url value") from None


def _check_response(response: bytes) -> None:
    if type(response) is not bytes or not 1 <= len(response) <= MAX_RESPONSE_BYTES:
        raise ValueError(f"a recorded response is between 1 and {MAX_RESPONSE_BYTES} bytes")
