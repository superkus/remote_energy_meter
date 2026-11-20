"""Restore scenarios across every durable store: a whole data tree rolled back to an older snapshot.

Each store is already tested alone. This pins how they behave *together* when an administrator restores an older backup
of the whole tree, with several secret generations in between: every record still verifies (it was authentic when it was
written), a one-sided restore is caught by comparing against the peer, a declared coordinated restore is held in
quarantine until each obligation is met, and nothing restored regains a lifetime it had already used up.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from remote_energy_meter.authenticated_store import StateAuthenticationKeyStore
from remote_energy_meter.control_authority import (
    AuthorityRefusal,
    AuthorityRefused,
    ControlAuthority,
    ControlAuthorityStore,
    RevocationScope,
)
from remote_energy_meter.control_identity import ControlIdentity, ControlIdentityStore
from remote_energy_meter.protocol_v1 import encode_base64url
from remote_energy_meter.reporter_pki import ReporterLeaf, ReporterPkiStore
from remote_energy_meter.restore_guard import (
    ControlComparison,
    PeerControlState,
    QuarantineStore,
    compare_control_state,
    evaluate_gate,
)

T0 = int(datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp())
REPORTER = encode_base64url(b"\x07" * 32, exact_length=32)


class Clock:
    def __init__(self) -> None:
        self.seconds = T0

    def __call__(self) -> datetime:
        return datetime.fromtimestamp(self.seconds, UTC)


def jwk() -> bytes:
    numbers = ec.generate_private_key(ec.SECP256R1()).public_key().public_numbers()
    member = {
        "crv": "P-256",
        "kty": "EC",
        "x": encode_base64url(numbers.x.to_bytes(32, "big"), exact_length=32),
        "y": encode_base64url(numbers.y.to_bytes(32, "big"), exact_length=32),
    }
    return json.dumps(member, separators=(",", ":"), sort_keys=True).encode("ascii")


def digest(n: int) -> bytes:
    return hashlib.sha256(bytes([n])).digest()


def ident(n: int) -> str:
    return encode_base64url(bytes([n]) * 16, exact_length=16)


@dataclass
class Tree:
    """Every durable store of one add-on instance, rooted in one directory, as `/data` would hold them."""

    root: Path
    clock: Clock

    def stores(
        self,
    ) -> tuple[
        StateAuthenticationKeyStore, ControlIdentityStore, ReporterPkiStore, ControlAuthorityStore, QuarantineStore
    ]:
        keys = StateAuthenticationKeyStore(self.root / "keys")
        return (
            keys,
            ControlIdentityStore(self.root / "control", keys=keys, clock=self.clock),
            ReporterPkiStore(self.root / "reporter-ca", self.root / "reporter-leaf", keys=keys, clock=self.clock),
            ControlAuthorityStore(self.root / "authority", keys=keys, clock=self.clock),
            QuarantineStore(self.root / "quarantine", keys=keys, clock=self.clock),
        )

    def pair(self, n: int, identity: ControlIdentity) -> ControlAuthority:
        """Authorize a fresh integration key under the next key generation, with credential ``n``."""
        authority = self.stores()[3]
        expires = self.clock.seconds + 500
        return authority.authorize_pairing(
            jwk=jwk(),
            spki_sha256=identity.spki_sha256,
            spki_generation=identity.spki_generation,
            key_generation=authority.load().next_key_generation,
            credential_digest=digest(n),
            credential_expires_at=expires,
            bootstrap_id=ident(n),
            request_hash=digest(100 + n),
            response=b'{"ok":true}',
            result_expires_at=expires,
        )

    def recover(self, n: int) -> ControlAuthority:
        authority = self.stores()[3]
        held = authority.load().authority
        assert held is not None
        expires = self.clock.seconds + 500
        return authority.revoke_for_recovery(
            scope=RevocationScope.CONTROL_KEY,
            key_thumbprint=held.thumbprint,
            key_generation=held.key_generation,
            spki_sha256=held.spki_sha256,
            spki_generation=held.spki_generation,
            recovery_generation=authority.load().next_recovery_generation,
            credential_digest=digest(n),
            credential_expires_at=expires,
            recovery_id=ident(n),
            request_hash=digest(100 + n),
            response=b'{"revoked":true}',
            result_expires_at=expires,
        )

    def snapshot(self, destination: Path) -> Path:
        shutil.copytree(self.root, destination)
        return destination

    def restore(self, snapshot: Path) -> None:
        shutil.rmtree(self.root)
        shutil.copytree(snapshot, self.root)


def peer_of(state: ControlAuthority, recovery: int | None = None) -> PeerControlState:
    held = state.authority
    assert held is not None
    return PeerControlState(
        key_thumbprint=held.thumbprint,
        key_generation=held.key_generation,
        spki_sha256=held.spki_sha256,
        spki_generation=held.spki_generation,
        recovery_generation=state.recovery_high_water if recovery is None else recovery,
    )


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    instance = Tree(tmp_path / "data", Clock())
    instance.root.mkdir()  # like /data, the parent of every store exists before the first store is created
    keys, identity_store, pki, _authority, _quarantine = instance.stores()
    keys.create()
    identity_store.create()
    pki.create(["ha.example.com"])
    return instance


def test_an_older_snapshot_of_the_whole_tree_still_verifies_but_is_one_generation_behind(
    tree: Tree, tmp_path: Path
) -> None:
    identity = tree.stores()[1].load()
    first = tree.pair(1, identity)
    snapshot_one = tree.snapshot(tmp_path / "snapshot-one")
    tree.clock.seconds += 300
    tree.recover(2)
    tree.clock.seconds += 10
    second = tree.pair(3, identity)
    assert (first.authority.key_generation, second.authority.key_generation) == (1, 2)  # type: ignore[union-attr]

    tree.restore(snapshot_one)

    _keys, identity_store, pki, authority, _quarantine = tree.stores()
    restored = authority.load()
    assert restored.authority is not None and restored.authority.key_generation == 1
    assert identity_store.load().spki_sha256 == identity.spki_sha256  # authentic, and the same SPKI as before
    assert pki.load_leaf().ca_spki_sha256 == pki.load_ca().spki_sha256
    # The integration, not restored, still holds generation 2: the one-sided restore is caught by the comparison.
    outcome = compare_control_state(restored, peer_of(second))
    assert outcome is ControlComparison.LOCAL_BEHIND and not outcome.control_enabled


def test_a_restored_integration_against_a_current_add_on_is_caught_from_the_other_side(
    tree: Tree, tmp_path: Path
) -> None:
    identity = tree.stores()[1].load()
    older_peer = peer_of(tree.pair(1, identity))
    tree.clock.seconds += 300
    tree.recover(2)
    tree.clock.seconds += 10
    current = tree.pair(3, identity)

    outcome = compare_control_state(current, older_peer)

    # The restored integration holds generation 1, which the recovery in between revoked.
    assert outcome is ControlComparison.PEER_REVOKED
    assert not outcome.control_enabled


def test_a_coordinated_restore_cannot_be_detected_so_the_declaration_holds_service_until_everything_is_redone(
    tree: Tree, tmp_path: Path
) -> None:
    identity = tree.stores()[1].load()
    first = tree.pair(1, identity)
    snapshot_one = tree.snapshot(tmp_path / "snapshot-one")
    tree.clock.seconds += 300
    tree.recover(2)
    tree.clock.seconds += 10
    tree.pair(3, identity)
    tree.restore(snapshot_one)
    restored = tree.stores()[3].load()
    # Both endpoints rolled back together, so the pair is consistent and nothing can tell.
    assert compare_control_state(restored, peer_of(first)) is ControlComparison.MATCH
    _keys, _identity, _pki, _authority, quarantine = tree.stores()
    assert evaluate_gate(quarantine).permitted  # an undeclared restore is not quarantined

    state = quarantine.declare_restore([REPORTER])

    assert not evaluate_gate(quarantine).permitted
    assert not state.control_administration_permitted and not state.reporter_may_operate(REPORTER)
    # Re-pairing: recover the restored authority, then pair a fresh key under the next generation.
    tree.clock.seconds += 20
    tree.recover(4)
    repaired = tree.pair(5, identity)
    assert repaired.authority is not None and repaired.authority.key_generation == 2  # never the restored 1
    assert quarantine.mark_control_repaired().control_administration_permitted
    quarantine.dispose_reporter(REPORTER)
    assert not evaluate_gate(quarantine).permitted  # MQTT v2 has not been re-provisioned yet
    quarantine.mark_mqtt_reprovisioned()

    assert quarantine.release().normal_service_permitted
    assert evaluate_gate(quarantine).permitted


def test_a_restored_credential_does_not_regain_the_lifetime_it_already_used(tree: Tree, tmp_path: Path) -> None:
    identity = tree.stores()[1].load()
    tree.pair(1, identity)
    snapshot_one = tree.snapshot(tmp_path / "snapshot-one")
    tree.restore(snapshot_one)
    tree.clock.seconds += 10_000  # long after the credential's embedded expiry
    authority = tree.stores()[3]
    tree.recover(2)  # a recovery write compacts the old digest away
    assert not authority.load().is_consumed(digest(1))

    with pytest.raises(AuthorityRefused) as caught:
        authority.authorize_pairing(
            jwk=jwk(),
            spki_sha256=identity.spki_sha256,
            spki_generation=identity.spki_generation,
            key_generation=authority.load().next_key_generation,
            credential_digest=digest(1),
            credential_expires_at=T0 + 500,  # the original embedded expiry
            bootstrap_id=ident(9),
            request_hash=digest(109),
            response=b"x",
            result_expires_at=T0 + 500,
        )

    assert caught.value.reason is AuthorityRefusal.CREDENTIAL_EXPIRED


def test_a_leaf_renewed_after_the_snapshot_is_older_but_still_chains_to_the_unchanged_ca(
    tree: Tree, tmp_path: Path
) -> None:
    _keys, _identity, pki, _authority, _quarantine = tree.stores()
    before: ReporterLeaf = pki.load_leaf()
    snapshot = tree.snapshot(tmp_path / "snapshot")
    tree.clock.seconds += 340 * 86_400
    renewed = tree.stores()[2].renew_leaf()
    assert renewed.spki_sha256 == before.spki_sha256

    tree.restore(snapshot)

    restored = tree.stores()[2].load_leaf()
    assert restored.certificate_der == before.certificate_der  # the older certificate came back, authentically
    assert restored.spki_sha256 == renewed.spki_sha256  # but the key and pin reporters rely on never changed
    assert restored.ca_spki_sha256 == renewed.ca_spki_sha256
