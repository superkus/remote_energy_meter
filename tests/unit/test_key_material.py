"""Shared P-256 key handling: pins, the redacted scalar holder, and timezone-aware times."""

from __future__ import annotations

import base64
import copy
import hashlib
import pickle
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from remote_energy_meter import key_material
from remote_energy_meter.key_material import (
    HASH_BYTES,
    SCALAR_BYTES,
    PrivateScalar,
    require_aware,
    scalar_bytes,
    spki_der,
    spki_identifier,
)


def test_the_pin_is_the_unpadded_base64url_sha256_of_the_der_spki() -> None:
    public_key = ec.generate_private_key(ec.SECP256R1()).public_key()
    der = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)

    assert spki_der(public_key) == der
    expected = base64.urlsafe_b64encode(hashlib.sha256(der).digest()).rstrip(b"=").decode("ascii")
    assert spki_identifier(public_key) == expected
    assert len(expected) == 43
    assert HASH_BYTES == 32


def test_the_scalar_round_trips_through_the_holder() -> None:
    private_key = ec.generate_private_key(ec.SECP256R1())
    raw = scalar_bytes(private_key)

    assert len(raw) == SCALAR_BYTES == 32
    assert PrivateScalar(raw).key().private_numbers() == private_key.private_numbers()


def test_a_scalar_with_leading_zero_bytes_is_still_32_bytes() -> None:
    private_key = ec.derive_private_key(5, ec.SECP256R1())

    assert scalar_bytes(private_key) == b"\x00" * 31 + b"\x05"


@pytest.mark.parametrize("value", [b"", b"\x01" * 31, b"\x01" * 33, bytearray(32), "x" * 32])
def test_the_holder_requires_exactly_32_bytes(value: object) -> None:
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        PrivateScalar(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, 2**256 - 1])
def test_an_out_of_range_scalar_is_refused_when_the_key_is_built(value: int) -> None:
    holder = PrivateScalar(value.to_bytes(32, "big"))

    with pytest.raises(ValueError):
        holder.key()


def test_the_holder_never_reveals_the_scalar() -> None:
    raw = b"\xab" * 32
    holder = PrivateScalar(raw)

    assert repr(holder) == str(holder) == "PrivateScalar(<redacted>)"
    assert raw.hex() not in repr(holder)
    for operation in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            operation(holder)


def test_a_naive_datetime_is_rejected_and_an_aware_one_returned_unchanged() -> None:
    aware = datetime(2030, 1, 1, tzinfo=UTC)

    assert require_aware(aware, "now") is aware
    with pytest.raises(ValueError, match="now must be a timezone-aware datetime"):
        require_aware(datetime(2030, 1, 1), "now")  # noqa: DTZ001


def test_the_module_exports_its_public_names() -> None:
    for name in key_material.__all__:
        assert hasattr(key_material, name)
