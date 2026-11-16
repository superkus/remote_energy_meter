"""The control authority: atomic pairing and recovery, one-use credentials, exact retries, and restore safety."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from remote_energy_meter import control_authority
from remote_energy_meter.authenticated_store import AuthenticatedDocumentStore, StateAuthenticationKeyStore
from remote_energy_meter.control_authority import (
    CREDENTIAL_MAX_LIFETIME_SECONDS,
    DOCUMENT_NAME,
    MAX_CONSUMED,
    MAX_RESPONSE_BYTES,
    MAX_RESULTS,
    TOMBSTONE_GRACE_SECONDS,
    AuthorityRefusal,
    AuthorityRefused,
    ConsumedCredential,
    ControlAuthority,
    ControlAuthorityStore,
    ExactResult,
    ResultKind,
    RevocationScope,
)
from remote_energy_meter.key_material import HASH_BYTES
from remote_energy_meter.protocol_v1 import encode_base64url
from remote_energy_meter.secure_store import SecureStoreError, StoreFault

T0 = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)
NOW = int(T0.timestamp())
SPKI = encode_base64url(b"\x11" * 32, exact_length=32)
OTHER_SPKI = encode_base64url(b"\x22" * 32, exact_length=32)


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def at(self, seconds: int) -> None:
        self.now = datetime.fromtimestamp(seconds, UTC)


def canonical_jwk() -> bytes:
    numbers = ec.generate_private_key(ec.SECP256R1()).public_key().public_numbers()
    member = {
        "crv": "P-256",
        "kty": "EC",
        "x": encode_base64url(numbers.x.to_bytes(32, "big"), exact_length=32),
        "y": encode_base64url(numbers.y.to_bytes(32, "big"), exact_length=32),
    }
    return json.dumps(member, separators=(",", ":"), sort_keys=True).encode("ascii")


def ident(n: int) -> str:
    return encode_base64url(bytes([n]) * 16, exact_length=16)


def digest(n: int) -> bytes:
    return hashlib.sha256(bytes([n])).digest()


@pytest.fixture
def keys(tmp_path: Path) -> StateAuthenticationKeyStore:
    store = StateAuthenticationKeyStore(tmp_path / "keys")
    store.create()
    return store


@pytest.fixture
def clock() -> Clock:
    return Clock(T0)


@pytest.fixture
def store(tmp_path: Path, keys: StateAuthenticationKeyStore, clock: Clock) -> ControlAuthorityStore:
    return ControlAuthorityStore(tmp_path / "authority", keys=keys, clock=clock)


def pair(store: ControlAuthorityStore, **overrides: Any) -> ControlAuthority:
    """Authorize a pairing with sensible defaults; ``overrides`` replace any argument."""
    arguments: dict[str, Any] = {
        "jwk": canonical_jwk(),
        "spki_sha256": SPKI,
        "spki_generation": 1,
        "key_generation": store.load().next_key_generation,
        "credential_digest": digest(1),
        "credential_expires_at": NOW + 600,
        "bootstrap_id": ident(1),
        "request_hash": digest(101),
        "response": b'{"ok":true}',
        "result_expires_at": NOW + 600,
    }
    arguments.update(overrides)
    return store.authorize_pairing(**arguments)


def recover(store: ControlAuthorityStore, **overrides: Any) -> ControlAuthority:
    """Revoke the current authority with defaults taken from the stored state."""
    current = store.load().authority
    assert current is not None
    arguments: dict[str, Any] = {
        "scope": RevocationScope.CONTROL_KEY,
        "key_thumbprint": current.thumbprint,
        "key_generation": current.key_generation,
        "spki_sha256": current.spki_sha256,
        "spki_generation": current.spki_generation,
        "recovery_generation": store.load().next_recovery_generation,
        "credential_digest": digest(2),
        "credential_expires_at": NOW + 600,
        "recovery_id": ident(2),
        "request_hash": digest(102),
        "response": b'{"revoked":true}',
        "result_expires_at": NOW + 600,
    }
    arguments.update(overrides)
    return store.revoke_for_recovery(**arguments)


def refusal_of(operation: Callable[[], object]) -> AuthorityRefusal:
    with pytest.raises(AuthorityRefused) as caught:
        operation()
    return caught.value.reason


def fault_of(operation: Callable[[], object]) -> StoreFault:
    with pytest.raises(SecureStoreError) as caught:
        operation()
    return caught.value.fault


def unchanged(store: ControlAuthorityStore, operation: Callable[[], object]) -> AuthorityRefusal:
    """Run a refused operation and prove the document is byte-identical afterwards."""
    before = store.path.read_bytes()
    reason = refusal_of(operation)
    assert store.path.read_bytes() == before
    return reason


# --------------------------------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------------------------------


def test_a_never_written_store_is_the_empty_authority_and_load_writes_nothing(store: ControlAuthorityStore) -> None:
    state = store.load()

    assert state.authority is None
    assert (state.key_high_water, state.key_revoked_through, state.spki_revoked_through) == (0, 0, 0)
    assert state.next_key_generation == 1
    assert state.next_recovery_generation == 1
    assert state.consumed == () and state.results == ()
    assert not store.path.exists()


def test_pairing_authorizes_the_key_consumes_the_credential_and_records_the_result_together(
    store: ControlAuthorityStore, tmp_path: Path, keys: StateAuthenticationKeyStore, clock: Clock
) -> None:
    jwk = canonical_jwk()
    state = pair(store, jwk=jwk)
    reloaded = ControlAuthorityStore(tmp_path / "authority", keys=keys, clock=clock).load()

    assert reloaded == state
    assert state.authority is not None
    assert state.authority.jwk == jwk
    assert (state.authority.key_generation, state.authority.spki_generation) == (1, 1)
    assert state.authority.spki_sha256 == SPKI
    assert state.authority.authorized_at == NOW
    assert state.key_high_water == 1
    assert state.is_consumed(digest(1))
    assert [(r.kind, r.identifier, r.response) for r in state.results] == [
        (ResultKind.BOOTSTRAP_CONFIRM, ident(1), b'{"ok":true}')
    ]
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_the_thumbprint_is_derived_from_the_key_not_taken_from_the_caller(store: ControlAuthorityStore) -> None:
    jwk = canonical_jwk()
    state = pair(store, jwk=jwk)

    from remote_energy_meter.protocol_v1 import parse_p256_public_jwk

    assert state.authority is not None
    assert state.authority.thumbprint == parse_p256_public_jwk(jwk).thumbprint


def test_a_second_pairing_while_paired_is_refused(store: ControlAuthorityStore) -> None:
    pair(store)

    reason = unchanged(
        store, lambda: pair(store, credential_digest=digest(7), bootstrap_id=ident(7), request_hash=digest(107))
    )

    assert reason is AuthorityRefusal.ALREADY_PAIRED


def test_a_consumed_credential_is_never_accepted_again(store: ControlAuthorityStore) -> None:
    pair(store)
    recover(store)

    reason = unchanged(store, lambda: pair(store, bootstrap_id=ident(9), request_hash=digest(109)))

    assert reason is AuthorityRefusal.CREDENTIAL_CONSUMED


@pytest.mark.parametrize(("offset", "refused"), [(-1, False), (0, True), (1, True)])
def test_a_credential_is_usable_only_strictly_before_its_embedded_expiry(
    store: ControlAuthorityStore, clock: Clock, offset: int, refused: bool
) -> None:
    clock.at(NOW + 600 + offset)

    if refused:
        assert unchanged_or_absent(store, lambda: pair(store)) is AuthorityRefusal.CREDENTIAL_EXPIRED
    else:
        assert pair(store).authority is not None


def unchanged_or_absent(store: ControlAuthorityStore, operation: Callable[[], object]) -> AuthorityRefusal:
    existed = store.path.exists()
    before = store.path.read_bytes() if existed else None
    reason = refusal_of(operation)
    assert store.path.exists() == existed
    assert (store.path.read_bytes() if existed else None) == before
    return reason


def test_the_key_generation_must_be_the_next_one(store: ControlAuthorityStore) -> None:
    for generation in (0, 2, 5):
        assert unchanged_or_absent(store, lambda g=generation: pair(store, key_generation=g)) is (  # type: ignore[misc]
            AuthorityRefusal.KEY_GENERATION_MISMATCH
        )
    assert pair(store, key_generation=1).authority is not None


def test_generations_are_never_reused_after_a_revocation(store: ControlAuthorityStore) -> None:
    pair(store)
    recover(store)

    assert store.load().next_key_generation == 2
    assert unchanged(
        store, lambda: pair(store, key_generation=1, credential_digest=digest(3), bootstrap_id=ident(3))
    ) is (AuthorityRefusal.KEY_GENERATION_MISMATCH)
    second = pair(store, credential_digest=digest(3), bootstrap_id=ident(3), request_hash=digest(103))
    assert second.authority is not None and second.authority.key_generation == 2


def test_a_result_identifier_cannot_be_reused(store: ControlAuthorityStore) -> None:
    pair(store)
    recover(store)

    reason = unchanged(store, lambda: pair(store, credential_digest=digest(3), request_hash=digest(103)))

    assert reason is AuthorityRefusal.RESULT_ID_REUSED


def test_the_same_identifier_may_be_used_by_the_other_result_kind(store: ControlAuthorityStore) -> None:
    pair(store, bootstrap_id=ident(5))

    state = recover(store, recovery_id=ident(5))

    assert {(r.kind, r.identifier) for r in state.results} == {
        (ResultKind.BOOTSTRAP_CONFIRM, ident(5)),
        (ResultKind.RECOVERY_REVOKE, ident(5)),
    }


# --------------------------------------------------------------------------------------------------
# exact results
# --------------------------------------------------------------------------------------------------


def test_an_exact_retry_returns_the_recorded_response_inside_the_window(store: ControlAuthorityStore) -> None:
    pair(store)

    lookup = store.lookup_result(ResultKind.BOOTSTRAP_CONFIRM, ident(1), digest(101))

    assert (lookup.response, lookup.conflict) == (b'{"ok":true}', False)


def test_a_changed_request_for_a_known_identifier_is_a_conflict_not_a_response(store: ControlAuthorityStore) -> None:
    pair(store)

    lookup = store.lookup_result(ResultKind.BOOTSTRAP_CONFIRM, ident(1), digest(150))

    assert (lookup.response, lookup.conflict) == (None, True)


def test_unknown_identifiers_and_the_other_kind_find_nothing(store: ControlAuthorityStore) -> None:
    pair(store)

    for kind, identifier in ((ResultKind.BOOTSTRAP_CONFIRM, ident(9)), (ResultKind.RECOVERY_REVOKE, ident(1))):
        lookup = store.lookup_result(kind, identifier, digest(101))
        assert (lookup.response, lookup.conflict) == (None, False)


@pytest.mark.parametrize(
    ("offset", "returnable"), [(-1, True), (0, False), (1, False), (TOMBSTONE_GRACE_SECONDS + 1, False)]
)
def test_a_result_is_returnable_only_strictly_before_its_expiry_whether_or_not_it_was_compacted(
    store: ControlAuthorityStore, clock: Clock, offset: int, returnable: bool
) -> None:
    pair(store)
    clock.at(NOW + 600 + offset)

    lookup = store.lookup_result(ResultKind.BOOTSTRAP_CONFIRM, ident(1), digest(101))

    assert (lookup.response is not None) is returnable
    assert lookup.conflict is False


def test_a_changed_request_after_the_window_is_not_a_conflict(store: ControlAuthorityStore, clock: Clock) -> None:
    pair(store)
    clock.at(NOW + 600)

    assert store.lookup_result(ResultKind.BOOTSTRAP_CONFIRM, ident(1), digest(150)).conflict is False


def test_lookup_validates_its_arguments_and_writes_nothing(store: ControlAuthorityStore) -> None:
    with pytest.raises(ValueError, match="identifier"):
        store.lookup_result(ResultKind.BOOTSTRAP_CONFIRM, "short", digest(1))
    with pytest.raises(ValueError, match="request_hash"):
        store.lookup_result(ResultKind.BOOTSTRAP_CONFIRM, ident(1), b"short")
    assert not store.path.exists()


# --------------------------------------------------------------------------------------------------
# recovery
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scope", "key_revoked", "spki_revoked"),
    [
        (RevocationScope.CONTROL_KEY, 1, 0),
        (RevocationScope.CONTROL_SPKI, 0, 1),
        (RevocationScope.CONTROL_KEY_AND_SPKI, 1, 1),
    ],
)
def test_recovery_revokes_exactly_the_named_generations_and_clears_the_authority(
    store: ControlAuthorityStore, scope: RevocationScope, key_revoked: int, spki_revoked: int
) -> None:
    pair(store)

    state = recover(store, scope=scope)

    assert state.authority is None
    assert (state.key_revoked_through, state.spki_revoked_through) == (key_revoked, spki_revoked)
    assert state.recovery_high_water == 1 and state.next_recovery_generation == 2
    assert state.is_consumed(digest(1)) and state.is_consumed(digest(2))
    assert {r.kind for r in state.results} == {ResultKind.BOOTSTRAP_CONFIRM, ResultKind.RECOVERY_REVOKE}
    lookup = store.lookup_result(ResultKind.RECOVERY_REVOKE, ident(2), digest(102))
    assert lookup.response == b'{"revoked":true}'


def test_a_key_only_recovery_allows_re_pairing_under_the_same_spki_generation(store: ControlAuthorityStore) -> None:
    pair(store)
    recover(store, scope=RevocationScope.CONTROL_KEY)

    state = pair(
        store,
        spki_generation=1,
        credential_digest=digest(3),
        bootstrap_id=ident(3),
        request_hash=digest(103),
    )

    assert state.authority is not None
    assert (state.authority.key_generation, state.authority.spki_generation) == (2, 1)


def test_an_spki_recovery_forbids_the_revoked_spki_generation_and_allows_a_newer_one(
    store: ControlAuthorityStore,
) -> None:
    pair(store)
    recover(store, scope=RevocationScope.CONTROL_SPKI)
    arguments = {"credential_digest": digest(3), "bootstrap_id": ident(3), "request_hash": digest(103)}

    assert unchanged(store, lambda: pair(store, spki_generation=1, **arguments)) is AuthorityRefusal.SPKI_REVOKED
    state = pair(store, spki_generation=2, **arguments)

    assert state.authority is not None and state.authority.spki_generation == 2


def test_recovery_without_authority_is_refused(store: ControlAuthorityStore) -> None:
    pair(store)
    current = store.load().authority
    assert current is not None
    recover(store)

    reason = unchanged(
        store,
        lambda: store.revoke_for_recovery(
            scope=RevocationScope.CONTROL_KEY,
            key_thumbprint=current.thumbprint,
            key_generation=1,
            spki_sha256=SPKI,
            spki_generation=1,
            recovery_generation=2,
            credential_digest=digest(4),
            credential_expires_at=NOW + 600,
            recovery_id=ident(4),
            request_hash=digest(104),
            response=b"x",
            result_expires_at=NOW + 600,
        ),
    )

    assert reason is AuthorityRefusal.NOT_PAIRED


def test_recovery_is_refused_as_exhausted_when_the_recovery_counter_is_at_its_maximum(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore
) -> None:
    pair(store)
    exhausted = forge(store, keys, recovery_high_water=UINT64_MAX)
    current = exhausted.load().authority
    assert current is not None

    reason = unchanged(
        exhausted,
        lambda: exhausted.revoke_for_recovery(
            scope=RevocationScope.CONTROL_KEY,
            key_thumbprint=current.thumbprint,
            key_generation=current.key_generation,
            spki_sha256=current.spki_sha256,
            spki_generation=current.spki_generation,
            recovery_generation=UINT64_MAX,
            credential_digest=digest(2),
            credential_expires_at=NOW + 600,
            recovery_id=ident(2),
            request_hash=digest(102),
            response=b"x",
            result_expires_at=NOW + 600,
        ),
    )

    assert reason is AuthorityRefusal.GENERATION_EXHAUSTED


@pytest.mark.parametrize(
    "mismatch",
    [
        {"key_thumbprint": encode_base64url(b"\x33" * 32, exact_length=32)},
        {"key_generation": 2},
        {"spki_sha256": OTHER_SPKI},
        {"spki_generation": 2},
    ],
)
def test_recovery_must_name_exactly_the_stored_current_state(
    store: ControlAuthorityStore, mismatch: dict[str, Any]
) -> None:
    pair(store)

    assert unchanged(store, lambda: recover(store, **mismatch)) is AuthorityRefusal.CURRENT_STATE_MISMATCH
    assert store.load().authority is not None


@pytest.mark.parametrize("generation", [0, 2, 3])
def test_recovery_must_name_the_next_recovery_generation(store: ControlAuthorityStore, generation: int) -> None:
    pair(store)

    assert unchanged(store, lambda: recover(store, recovery_generation=generation)) is (
        AuthorityRefusal.RECOVERY_GENERATION_MISMATCH
    )


def test_recovery_consumes_a_fresh_credential_only(store: ControlAuthorityStore, clock: Clock) -> None:
    pair(store)

    assert unchanged(store, lambda: recover(store, credential_digest=digest(1))) is AuthorityRefusal.CREDENTIAL_CONSUMED
    assert unchanged(store, lambda: recover(store, credential_expires_at=NOW)) is AuthorityRefusal.CREDENTIAL_EXPIRED
    assert recover(store).authority is None


def test_recovery_results_cannot_reuse_an_identifier(store: ControlAuthorityStore) -> None:
    pair(store)
    recover(store, recovery_id=ident(6))
    pair(store, credential_digest=digest(3), bootstrap_id=ident(3), request_hash=digest(103))

    reason = unchanged(
        store, lambda: recover(store, recovery_id=ident(6), credential_digest=digest(4), request_hash=digest(106))
    )

    assert reason is AuthorityRefusal.RESULT_ID_REUSED


# --------------------------------------------------------------------------------------------------
# compaction and restore
# --------------------------------------------------------------------------------------------------


def test_compaction_drops_a_consumed_digest_exactly_when_its_grace_ends() -> None:
    state = ControlAuthority(None, 1, 0, 0, 0, (ConsumedCredential(digest(1), NOW + 600),), ())
    boundary = NOW + 600 + TOMBSTONE_GRACE_SECONDS

    assert ControlAuthorityStore._compact(state, boundary - 1).consumed == state.consumed
    assert ControlAuthorityStore._compact(state, boundary).consumed == ()


def test_compaction_drops_a_result_body_at_its_expiry_and_the_tombstone_at_the_end_of_its_grace() -> None:
    result = ExactResult(ResultKind.BOOTSTRAP_CONFIRM, ident(1), digest(101), b"body", NOW + 600)
    state = ControlAuthority(None, 1, 0, 0, 0, (), (result,))
    compact = ControlAuthorityStore._compact

    assert compact(state, NOW + 599).results == (result,)
    assert compact(state, NOW + 600).results == (replace(result, response=None),)
    assert compact(state, NOW + 600 + TOMBSTONE_GRACE_SECONDS - 1).results == (replace(result, response=None),)
    assert compact(state, NOW + 600 + TOMBSTONE_GRACE_SECONDS).results == ()


def test_a_write_compacts_what_the_clock_has_made_unnecessary(store: ControlAuthorityStore, clock: Clock) -> None:
    pair(store)
    clock.at(NOW + 300)  # a credential lives at most ten minutes, so a later expiry needs a later creation
    recover(store, credential_expires_at=NOW + 900, result_expires_at=NOW + 900)
    clock.at(NOW + 600 + TOMBSTONE_GRACE_SECONDS)

    state = pair(
        store,
        credential_digest=digest(3),
        bootstrap_id=ident(3),
        request_hash=digest(103),
        credential_expires_at=NOW + 1500,
        result_expires_at=NOW + 1500,
    )

    assert not state.is_consumed(digest(1))  # expired at NOW+600, grace over
    assert state.is_consumed(digest(2))  # expires at NOW+900, still inside its grace
    assert {r.identifier for r in state.results} == {ident(2), ident(3)}  # the confirm tombstone is gone


def test_a_compacted_digest_does_not_make_an_expired_credential_usable_again(
    store: ControlAuthorityStore, clock: Clock
) -> None:
    pair(store)
    recover(store)
    clock.at(NOW + 600 + TOMBSTONE_GRACE_SECONDS)
    pair(
        store,
        credential_digest=digest(3),
        bootstrap_id=ident(3),
        request_hash=digest(103),
        credential_expires_at=NOW + 1500,
        result_expires_at=NOW + 1500,
    )
    assert not store.load().is_consumed(digest(1))
    recover(
        store,
        credential_digest=digest(4),
        recovery_id=ident(4),
        request_hash=digest(104),
        credential_expires_at=NOW + 1500,
        result_expires_at=NOW + 1500,
    )

    reason = unchanged(
        store, lambda: pair(store, credential_digest=digest(1), bootstrap_id=ident(8), request_hash=digest(108))
    )

    assert reason is AuthorityRefusal.CREDENTIAL_EXPIRED


def test_restoring_an_older_backup_never_extends_an_embedded_expiry(
    store: ControlAuthorityStore, tmp_path: Path, keys: StateAuthenticationKeyStore, clock: Clock
) -> None:
    pair(store)
    backup = store.path.read_bytes()
    recover(store)
    # The administrator restores the backup taken right after pairing, long after the credentials expired.
    store.path.write_bytes(backup)
    store.path.chmod(0o600)
    clock.at(NOW + 10_000)
    restored = ControlAuthorityStore(tmp_path / "authority", keys=keys, clock=clock)

    state = restored.load()

    assert state.authority is not None  # a restored pairing is authentic; the quarantine work owns that case
    assert [c.expires_at for c in state.consumed] == [NOW + 600]  # exactly the recorded instant, not re-derived
    assert restored.lookup_result(ResultKind.BOOTSTRAP_CONFIRM, ident(1), digest(101)).response is None


def test_a_restored_state_without_the_consumed_digest_still_refuses_the_expired_credential(
    store: ControlAuthorityStore, tmp_path: Path, keys: StateAuthenticationKeyStore, clock: Clock
) -> None:
    empty_backup_time = NOW
    pair(store, credential_digest=digest(1))
    recover(store)
    # An older backup that predates the pairing has no record of the digest at all; restore it by removing the file.
    store.path.unlink()
    clock.at(empty_backup_time + 601)
    fresh = ControlAuthorityStore(tmp_path / "authority", keys=keys, clock=clock)

    assert unchanged_or_absent(fresh, lambda: pair(fresh)) is AuthorityRefusal.CREDENTIAL_EXPIRED


# --------------------------------------------------------------------------------------------------
# atomicity, concurrency, capacity
# --------------------------------------------------------------------------------------------------


def test_a_failed_write_leaves_the_credential_unconsumed_and_no_authority(
    store: ControlAuthorityStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "replace", refuse)

    assert fault_of(lambda: pair(store)) is StoreFault.IO_ERROR
    monkeypatch.undo()
    assert store.load().authority is None
    assert not store.load().is_consumed(digest(1))
    assert pair(store).authority is not None


def test_two_threads_pairing_with_one_credential_succeed_exactly_once(store: ControlAuthorityStore) -> None:
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt(index: int) -> None:
        try:
            pair(
                store,
                jwk=canonical_jwk(),
                bootstrap_id=ident(index),
                request_hash=digest(100 + index),
                key_generation=1,
            )
            result = "paired"
        except AuthorityRefused as refused:
            result = refused.reason.value
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt, args=(index,)) for index in range(1, 7)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("paired") == 1
    assert set(outcomes) <= {"paired", "already_paired"}
    assert store.load().authority is not None


def test_the_consumed_credential_capacity_is_enforced_before_anything_is_consumed(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore
) -> None:
    full = [
        {"digest": encode_base64url(digest(i), exact_length=32), "expires_at": NOW + 1000} for i in range(MAX_CONSUMED)
    ]
    fresh = at_limit(store, keys, consumed=full)

    assert unchanged(fresh, lambda: pair(fresh, credential_digest=digest(200))) is AuthorityRefusal.CAPACITY_EXCEEDED
    assert len(fresh.load().consumed) == MAX_CONSUMED


def test_the_recorded_result_capacity_is_enforced_before_anything_is_consumed(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore
) -> None:
    full = [
        {
            "expires_at": NOW + 1000,
            "id": ident(index),
            "kind": "bootstrap-confirm",
            "request_hash": encode_base64url(digest(index), exact_length=32),
            "response": encode_base64url(b"r", exact_length=1),
        }
        for index in range(MAX_RESULTS)
    ]
    fresh = at_limit(store, keys, results=full)

    reason = unchanged(fresh, lambda: pair(fresh, credential_digest=digest(200), bootstrap_id=ident(200)))

    assert reason is AuthorityRefusal.CAPACITY_EXCEEDED
    assert not fresh.load().is_consumed(digest(200))


def at_limit(store: ControlAuthorityStore, keys: StateAuthenticationKeyStore, **changes: Any) -> ControlAuthorityStore:
    """A store whose authentic record holds ``changes`` (for example a full consumed list), read afresh."""
    pair(store)
    recover(store)
    return forge(store, keys, **changes)


# --------------------------------------------------------------------------------------------------
# tampering and invariants
# --------------------------------------------------------------------------------------------------


def second_writer(store: ControlAuthorityStore, keys: StateAuthenticationKeyStore) -> AuthenticatedDocumentStore:
    return AuthenticatedDocumentStore(
        store.path.parent,
        DOCUMENT_NAME,
        store_id="control-authority",
        schema_version=1,
        max_bytes=256 * 1024,
        keys=keys,
    )


def forge(store: ControlAuthorityStore, keys: StateAuthenticationKeyStore, **changes: Any) -> ControlAuthorityStore:
    """Re-save the record correctly tagged but altered; return a fresh store reading it, as after a restart."""
    document = second_writer(store, keys)
    loaded = document.load()
    assert loaded is not None
    document.save({**loaded.record, **changes})
    return ControlAuthorityStore(store.path.parent, keys=keys, clock=store._clock)


def test_an_edited_record_fails_authentication(store: ControlAuthorityStore, keys: StateAuthenticationKeyStore) -> None:
    pair(store)
    store.path.write_bytes(store.path.read_bytes().replace(b'"key_high_water":1', b'"key_high_water":9'))

    fresh = ControlAuthorityStore(store.path.parent, keys=keys, clock=store._clock)
    assert fault_of(fresh.load) is StoreFault.AUTHENTICATION_FAILED


def test_an_authentic_record_that_breaks_an_invariant_is_inconsistent(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore
) -> None:
    pair(store)
    document = second_writer(store, keys).load()
    assert document is not None
    record: dict[str, Any] = dict(document.record)
    authority = dict(record["authority"])
    cases: list[dict[str, Any]] = [
        {"key_revoked_through": 5},
        {"key_revoked_through": 1},
        {"spki_revoked_through": 1},
        {"consumed": [record["consumed"][0], record["consumed"][0]]},
        {"results": [record["results"][0], record["results"][0]]},
        {"authority": {**authority, "key_generation": 2}},
    ]
    for changes in cases:
        fresh = forge(store, keys, **changes)
        assert fault_of(fresh.load) is StoreFault.IDENTITY_INCONSISTENT, changes
        forge(store, keys, **{key: record[key] for key in changes})  # restore for the next case


@pytest.mark.parametrize(
    "changes",
    [
        {"latest_seen": "1"},
        {"latest_seen": None},
        {"key_high_water": "1"},
        {"key_high_water": True},
        {"recovery_high_water": None},
        {"consumed": "no"},
        {"consumed": [{"digest": "x", "expires_at": 1}]},
        {"consumed": [{"digest": encode_base64url(b"\x00" * 32, exact_length=32)}]},
        {"results": [{"id": "x"}]},
        {"authority": {"jwk": "x"}},
        {"authority": 5},
        {"extra": 1},
    ],
)
def test_a_record_of_the_wrong_shape_is_malformed(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore, changes: dict[str, Any]
) -> None:
    pair(store)

    assert fault_of(forge(store, keys, **changes).load) is StoreFault.MALFORMED


def test_a_thumbprint_that_is_not_the_hash_of_the_key_is_malformed(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore
) -> None:
    pair(store)
    document = second_writer(store, keys).load()
    assert document is not None
    authority = {**document.record["authority"], "thumbprint": encode_base64url(b"\x44" * 32, exact_length=32)}  # type: ignore[dict-item]

    assert fault_of(forge(store, keys, authority=authority).load) is StoreFault.MALFORMED


def test_the_record_never_contains_a_raw_credential_or_the_pairing_key(store: ControlAuthorityStore) -> None:
    pair(store)

    stored = store.path.read_bytes()

    assert b"remcp1" not in stored
    assert encode_base64url(digest(1), exact_length=32).encode() in stored  # only the digest is kept


# --------------------------------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"jwk": b"not a jwk"}, "JWK"),
        ({"credential_digest": b"short"}, "credential_digest"),
        ({"request_hash": b"short"}, "request_hash"),
        ({"bootstrap_id": "short"}, "bootstrap_id"),
        ({"spki_sha256": "short"}, "spki_sha256"),
        ({"spki_generation": -1}, "spki_generation"),
        ({"key_generation": 2**64}, "key_generation"),
        ({"credential_expires_at": True}, "credential_expires_at"),
        ({"result_expires_at": -5}, "result_expires_at"),
        ({"response": b""}, "response"),
        ({"response": b"x" * (MAX_RESPONSE_BYTES + 1)}, "response"),
    ],
)
def test_malformed_pairing_arguments_raise_value_error_and_write_nothing(
    store: ControlAuthorityStore, override: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pair(store, **override)
    assert not store.path.exists()


# --------------------------------------------------------------------------------------------------
# lifetimes: credential at most ten minutes, result no later than its credential
# --------------------------------------------------------------------------------------------------

LIFETIME_MESSAGE = "more than the maximum"
WINDOW_MESSAGE = "after now"
UINT64_MAX = 2**64 - 1


def directory_listing(store: ControlAuthorityStore) -> list[str]:
    return sorted(p.name for p in store.path.parent.iterdir()) if store.path.parent.exists() else []


def rejected_and_unchanged(store: ControlAuthorityStore, operation: Callable[[], object], message: str) -> None:
    """The operation raises ``ValueError`` matching ``message``; the document and its directory are as they were."""
    existed = store.path.exists()
    before = store.path.read_bytes() if existed else None
    files_before = directory_listing(store)
    with pytest.raises(ValueError, match=message):
        operation()
    assert store.path.exists() == existed
    assert (store.path.read_bytes() if existed else None) == before
    assert directory_listing(store) == files_before


def pairing(store: ControlAuthorityStore) -> Callable[..., ControlAuthority]:
    return lambda **overrides: pair(store, **overrides)


def recovery(store: ControlAuthorityStore) -> Callable[..., ControlAuthority]:
    pair(store)
    return lambda **overrides: recover(store, **overrides)


OPERATIONS = [pytest.param(pairing, id="pairing"), pytest.param(recovery, id="recovery")]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "result_expires_at",
    [NOW - 1, NOW, NOW + 601],
    ids=["before-now", "equal-to-now", "after-credential-expiry"],
)
def test_a_result_must_expire_after_now_and_no_later_than_its_credential(
    store: ControlAuthorityStore,
    operation: Callable[[ControlAuthorityStore], Callable[..., ControlAuthority]],
    result_expires_at: int,
) -> None:
    run = operation(store)

    rejected_and_unchanged(
        store, lambda: run(credential_expires_at=NOW + 600, result_expires_at=result_expires_at), WINDOW_MESSAGE
    )


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "result_expires_at", [NOW + 1, NOW + 300, NOW + 600], ids=["now+1", "inside", "equal-to-credential"]
)
def test_a_result_expiring_after_now_up_to_its_credential_expiry_is_accepted(
    store: ControlAuthorityStore,
    operation: Callable[[ControlAuthorityStore], Callable[..., ControlAuthority]],
    result_expires_at: int,
) -> None:
    run = operation(store)

    state = run(credential_expires_at=NOW + 600, result_expires_at=result_expires_at)

    assert result_expires_at in {r.expires_at for r in state.results}


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("credential_expires_at", [NOW + 601, NOW + 3600, UINT64_MAX])
def test_a_credential_may_not_expire_more_than_ten_minutes_after_now(
    store: ControlAuthorityStore,
    operation: Callable[[ControlAuthorityStore], Callable[..., ControlAuthority]],
    credential_expires_at: int,
) -> None:
    run = operation(store)

    rejected_and_unchanged(
        store,
        lambda: run(
            credential_expires_at=credential_expires_at, result_expires_at=NOW + CREDENTIAL_MAX_LIFETIME_SECONDS
        ),
        LIFETIME_MESSAGE,
    )


@pytest.mark.parametrize("operation", OPERATIONS)
def test_a_credential_expiring_exactly_ten_minutes_after_now_is_accepted(
    store: ControlAuthorityStore, operation: Callable[[ControlAuthorityStore], Callable[..., ControlAuthority]]
) -> None:
    run = operation(store)

    state = run(credential_expires_at=NOW + 600, result_expires_at=NOW + 600)

    assert any(c.expires_at == NOW + 600 for c in state.consumed)


def test_the_lifetime_bound_follows_the_injected_clock(store: ControlAuthorityStore, clock: Clock) -> None:
    clock.at(NOW + 1000)

    rejected_and_unchanged(
        store, lambda: pair(store, credential_expires_at=NOW + 1601, result_expires_at=NOW + 1600), LIFETIME_MESSAGE
    )
    assert pair(store, credential_expires_at=NOW + 1600, result_expires_at=NOW + 1600).authority is not None


@pytest.mark.parametrize("operation", OPERATIONS)
def test_the_minimal_window_of_one_second_is_accepted(
    store: ControlAuthorityStore, operation: Callable[[ControlAuthorityStore], Callable[..., ControlAuthority]]
) -> None:
    run = operation(store)

    state = run(credential_expires_at=NOW + 1, result_expires_at=NOW + 1)

    assert any(c.expires_at == NOW + 1 for c in state.consumed)
    assert any(r.expires_at == NOW + 1 for r in state.results)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_a_credential_expiring_exactly_now_is_expired_not_malformed(
    store: ControlAuthorityStore, operation: Callable[[ControlAuthorityStore], Callable[..., ControlAuthority]]
) -> None:
    run = operation(store)

    reason = unchanged_or_absent(store, lambda: run(credential_expires_at=NOW, result_expires_at=NOW + 1))

    assert reason is AuthorityRefusal.CREDENTIAL_EXPIRED


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("result_expires_at", [NOW - 5, NOW + 500], ids=["before-now", "after-credential-expiry"])
def test_an_expired_credential_is_refused_as_expired_before_its_result_window_is_judged(
    store: ControlAuthorityStore,
    operation: Callable[[ControlAuthorityStore], Callable[..., ControlAuthority]],
    result_expires_at: int,
) -> None:
    run = operation(store)

    reason = unchanged_or_absent(store, lambda: run(credential_expires_at=NOW - 1, result_expires_at=result_expires_at))

    assert reason is AuthorityRefusal.CREDENTIAL_EXPIRED


@pytest.mark.parametrize("operation", OPERATIONS)
def test_an_over_long_expiry_is_a_value_error_even_for_a_consumed_credential(
    store: ControlAuthorityStore, operation: Callable[[ControlAuthorityStore], Callable[..., ControlAuthority]]
) -> None:
    if operation is pairing:
        pair(store)  # consumes digest(1), then recovery leaves the store unpaired so a pairing reaches the rules
        recover(store)
    run = operation(store)  # the recovery operation pairs with digest(1) itself
    assert store.load().is_consumed(digest(1))

    rejected_and_unchanged(
        store,
        lambda: run(
            credential_digest=digest(1),
            credential_expires_at=NOW + CREDENTIAL_MAX_LIFETIME_SECONDS + 1,
            result_expires_at=NOW + CREDENTIAL_MAX_LIFETIME_SECONDS,
        ),
        LIFETIME_MESSAGE,
    )


@pytest.mark.parametrize("operation", OPERATIONS)
def test_after_a_rejected_lifetime_the_next_valid_operation_succeeds_and_state_is_unchanged(
    store: ControlAuthorityStore, operation: Callable[[ControlAuthorityStore], Callable[..., ControlAuthority]]
) -> None:
    run = operation(store)
    before = store.load()

    rejected_and_unchanged(
        store, lambda: run(credential_expires_at=NOW + CREDENTIAL_MAX_LIFETIME_SECONDS + 1), LIFETIME_MESSAGE
    )
    assert store.load() == before
    state = run()

    assert state.consumed != before.consumed


def test_a_document_at_every_capacity_with_maximal_bodies_fits_the_size_limit_and_round_trips(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore
) -> None:
    body = b"x" * MAX_RESPONSE_BYTES
    results = [
        {**r, "response": encode_base64url(body, exact_length=MAX_RESPONSE_BYTES)}
        for r in full_results(NOW + 1000)[: MAX_RESULTS - 1]
    ]
    fresh = at_limit(store, keys, consumed=full_consumed(NOW + 1000)[: MAX_CONSUMED - 1], results=results)

    state = pair(fresh, credential_digest=digest(200), bootstrap_id=ident(200), request_hash=digest(201), response=body)

    assert len(state.consumed) == MAX_CONSUMED and len(state.results) == MAX_RESULTS
    assert all(r.response == body for r in state.results)
    assert fresh.path.stat().st_size < control_authority._MAX_BYTES
    assert fresh.load() == state


# --------------------------------------------------------------------------------------------------
# capacity counts only live entries
# --------------------------------------------------------------------------------------------------


def full_consumed(expires_at: int) -> list[dict[str, Any]]:
    return [
        {"digest": encode_base64url(digest(i), exact_length=HASH_BYTES), "expires_at": expires_at}
        for i in range(MAX_CONSUMED)
    ]


def full_results(expires_at: int) -> list[dict[str, Any]]:
    return [
        {
            "expires_at": expires_at,
            "id": ident(index),
            "kind": "bootstrap-confirm",
            "request_hash": encode_base64url(digest(index), exact_length=HASH_BYTES),
            "response": encode_base64url(b"r", exact_length=1),
        }
        for index in range(MAX_RESULTS)
    ]


def test_expired_entries_at_capacity_do_not_block_a_pairing_or_a_recovery(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore, clock: Clock
) -> None:
    expiry = NOW + 1000
    fresh = at_limit(store, keys, consumed=full_consumed(expiry), results=full_results(expiry))
    clock.at(expiry + TOMBSTONE_GRACE_SECONDS + 1)
    now = expiry + TOMBSTONE_GRACE_SECONDS + 1

    paired = pair(
        fresh,
        credential_digest=digest(200),
        bootstrap_id=ident(200),
        request_hash=digest(201),
        credential_expires_at=now + 600,
        result_expires_at=now + 600,
    )
    recovered = recover(
        fresh,
        credential_digest=digest(202),
        recovery_id=ident(202),
        request_hash=digest(203),
        credential_expires_at=now + 600,
        result_expires_at=now + 600,
    )

    assert paired.authority is not None
    assert recovered.authority is None
    assert len(recovered.consumed) == 2 and len(recovered.results) == 2


@pytest.mark.parametrize("filled", ["consumed", "results"])
def test_live_entries_at_capacity_still_refuse_a_pairing_and_change_nothing(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore, filled: str
) -> None:
    changes = {"consumed": full_consumed(NOW + 1000)} if filled == "consumed" else {"results": full_results(NOW + 1000)}
    fresh = at_limit(store, keys, **changes)

    reason = unchanged(fresh, lambda: pair(fresh, credential_digest=digest(200), bootstrap_id=ident(200)))

    assert reason is AuthorityRefusal.CAPACITY_EXCEEDED


@pytest.mark.parametrize("filled", ["consumed", "results"])
def test_live_entries_at_capacity_still_refuse_a_recovery_and_change_nothing(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore, filled: str
) -> None:
    pair(store)
    changes = {"consumed": full_consumed(NOW + 1000)} if filled == "consumed" else {"results": full_results(NOW + 1000)}
    fresh = forge(store, keys, **changes)

    reason = unchanged(fresh, lambda: recover(fresh, credential_digest=digest(200), recovery_id=ident(200)))

    assert reason is AuthorityRefusal.CAPACITY_EXCEEDED
    assert fresh.load().authority is not None


# --------------------------------------------------------------------------------------------------
# the clock is read under the store lock, inside the update
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("operation", OPERATIONS)
def test_the_clock_is_read_inside_the_document_update_while_the_lock_is_held(
    tmp_path: Path,
    keys: StateAuthenticationKeyStore,
    operation: Callable[[ControlAuthorityStore], Callable[..., ControlAuthority]],
) -> None:
    events: list[str] = []
    holder: list[ControlAuthorityStore] = []

    def clock() -> datetime:
        # The private lock is read because it is the only observable for "the store lock is held".
        locked = bool(holder) and holder[0]._lock.locked()
        events.append(("clock-in-update" if "update" in events else "clock-locked") if locked else "clock-free")
        return T0

    store = ControlAuthorityStore(tmp_path / "authority", keys=keys, clock=clock)
    holder.append(store)
    run = operation(store)
    events.clear()
    real_update = store._document.update

    def recording_update(build: Callable[..., Any]) -> Any:
        def recording_build(current: Any) -> Any:
            events.append("update")
            return build(current)

        return real_update(recording_build)

    store._document.update = recording_update  # type: ignore[method-assign]

    run()

    assert "clock-free" not in events
    assert "clock-in-update" in events


def test_a_naive_clock_is_rejected(tmp_path: Path, keys: StateAuthenticationKeyStore) -> None:
    naive = ControlAuthorityStore(tmp_path / "authority", keys=keys, clock=lambda: datetime(2030, 1, 1))  # noqa: DTZ001

    with pytest.raises(ValueError, match="timezone-aware"):
        pair(naive)


def test_the_capacity_constants_are_bounded() -> None:
    assert (MAX_CONSUMED, MAX_RESULTS, MAX_RESPONSE_BYTES) == (64, 32, 2048)
    assert timedelta(minutes=5).seconds == TOMBSTONE_GRACE_SECONDS


def test_the_module_exports_its_public_names() -> None:
    for name in control_authority.__all__:
        assert hasattr(control_authority, name)


# --------------------------------------------------------------------------------------------------
# the clock guard
# --------------------------------------------------------------------------------------------------


def test_every_write_stores_the_time_it_acted_on(store: ControlAuthorityStore, clock: Clock) -> None:
    assert store.load().latest_seen == 0

    state = pair(store)
    assert state.latest_seen == NOW
    assert state.authority is not None and state.authority.authorized_at == NOW

    clock.at(NOW + 300)
    assert recover(store).latest_seen == NOW + 300


def test_time_never_goes_backwards_for_the_store(store: ControlAuthorityStore, clock: Clock) -> None:
    pair(store)
    clock.at(NOW + 300)
    recover(store)
    clock.at(NOW + 50)  # the wall clock stepped back by five minutes

    state = pair(
        store,
        credential_digest=digest(3),
        bootstrap_id=ident(3),
        request_hash=digest(103),
        credential_expires_at=NOW + 600,
        result_expires_at=NOW + 600,
    )

    assert state.latest_seen == NOW + 300  # not lowered
    assert state.authority is not None and state.authority.authorized_at == NOW + 300  # acted on the later time


def test_a_clock_that_steps_back_cannot_revive_a_compacted_credential(
    store: ControlAuthorityStore, clock: Clock
) -> None:
    pair(store)  # consumes digest 1, which expires at NOW+600
    clock.at(NOW + 300)
    recover(store, credential_expires_at=NOW + 900, result_expires_at=NOW + 900)
    clock.at(NOW + 600 + TOMBSTONE_GRACE_SECONDS)  # digest 1's grace ends: the next write compacts it
    pair(
        store,
        credential_digest=digest(3),
        bootstrap_id=ident(3),
        request_hash=digest(103),
        credential_expires_at=NOW + 1500,
        result_expires_at=NOW + 1500,
    )
    recover(
        store,
        credential_digest=digest(4),
        recovery_id=ident(4),
        request_hash=digest(104),
        credential_expires_at=NOW + 1500,
        result_expires_at=NOW + 1500,
    )
    assert not store.load().is_consumed(digest(1))  # compacted away
    clock.at(NOW + 100)  # the wall clock steps back far beyond the five-minute grace

    reason = unchanged(
        store, lambda: pair(store, credential_digest=digest(1), bootstrap_id=ident(8), request_hash=digest(108))
    )

    assert reason is AuthorityRefusal.CREDENTIAL_EXPIRED


def test_a_clock_that_steps_back_cannot_reopen_an_expired_result_window(
    store: ControlAuthorityStore, clock: Clock
) -> None:
    pair(store)  # the confirm result is returnable until NOW+600
    clock.at(NOW + 700)
    recover(store, credential_expires_at=NOW + 1000, result_expires_at=NOW + 1000)
    clock.at(NOW + 10)

    lookup = store.lookup_result(ResultKind.BOOTSTRAP_CONFIRM, ident(1), digest(101))

    assert (lookup.response, lookup.conflict) == (None, False)


def test_a_lookup_also_refuses_a_body_whose_window_the_stored_time_has_passed(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore, clock: Clock
) -> None:
    """Defence in depth: every write already blanks an expired body, so only an inconsistent record shows this."""
    pair(store)
    ahead = forge(store, keys, latest_seen=NOW + 700)
    clock.at(NOW + 10)

    lookup = ahead.lookup_result(ResultKind.BOOTSTRAP_CONFIRM, ident(1), digest(101))

    assert (lookup.response, lookup.conflict) == (None, False)


def test_while_the_store_is_ahead_of_the_clock_it_acts_on_the_later_time(
    store: ControlAuthorityStore, clock: Clock
) -> None:
    pair(store)
    clock.at(NOW + 300)
    recover(store)
    clock.at(NOW + 100)  # behind the stored time

    # A credential that expires before the stored time is refused, failing closed until real time catches up.
    refused = unchanged(
        store,
        lambda: pair(
            store,
            credential_digest=digest(3),
            bootstrap_id=ident(3),
            request_hash=digest(103),
            credential_expires_at=NOW + 250,
            result_expires_at=NOW + 250,
        ),
    )
    accepted = pair(
        store,
        credential_digest=digest(4),
        bootstrap_id=ident(4),
        request_hash=digest(104),
        credential_expires_at=NOW + 900,
        result_expires_at=NOW + 900,
    )

    assert refused is AuthorityRefusal.CREDENTIAL_EXPIRED
    assert accepted.authority is not None and accepted.latest_seen == NOW + 300


def test_a_record_authorizing_a_key_after_its_latest_time_is_inconsistent(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore
) -> None:
    pair(store)

    assert fault_of(forge(store, keys, latest_seen=NOW - 1).load) is StoreFault.IDENTITY_INCONSISTENT


def test_a_record_without_the_guard_field_is_malformed(
    store: ControlAuthorityStore, keys: StateAuthenticationKeyStore
) -> None:
    pair(store)
    document = second_writer(store, keys)
    loaded = document.load()
    assert loaded is not None
    document.save({key: value for key, value in loaded.record.items() if key != "latest_seen"})

    fresh = ControlAuthorityStore(store.path.parent, keys=keys, clock=store._clock)
    assert fault_of(fresh.load) is StoreFault.MALFORMED
