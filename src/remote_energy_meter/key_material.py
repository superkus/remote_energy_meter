"""Shared handling of P-256 key material for the add-on's durable identities.

Both the control TLS identity and the reporter PKI keep a P-256 private key as its 32-byte scalar inside an
authenticated record, and identify a key by the unpadded base64url SHA-256 of its DER SubjectPublicKeyInfo (SPKI).
This module holds the pieces they share so each identity does not carry its own copy:

* :func:`spki_der` and :func:`spki_identifier` derive that identifier.
* :class:`PrivateScalar` holds a private scalar so it never appears in ``repr``, ``str``, copies, or pickles.
* :func:`require_aware` rejects a naive ``datetime``, which Python would otherwise read as host-local time.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from remote_energy_meter.protocol_v1 import encode_base64url

__all__ = [
    "HASH_BYTES",
    "SCALAR_BYTES",
    "PrivateScalar",
    "require_aware",
    "scalar_bytes",
    "spki_der",
    "spki_identifier",
]

SCALAR_BYTES: Final = 32
"""Length of a P-256 private scalar."""

HASH_BYTES: Final = 32
"""Length of a SHA-256 digest."""


def spki_der(public_key: ec.EllipticCurvePublicKey) -> bytes:
    """Return the exact DER SubjectPublicKeyInfo of ``public_key``."""
    return public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)


def spki_identifier(public_key: ec.EllipticCurvePublicKey) -> str:
    """Return the canonical unpadded base64url SHA-256 of the DER SPKI: the pin peers verify."""
    return encode_base64url(hashlib.sha256(spki_der(public_key)).digest(), exact_length=HASH_BYTES)


def scalar_bytes(private_key: ec.EllipticCurvePrivateKey) -> bytes:
    """Return the private scalar as exactly 32 big-endian bytes."""
    return private_key.private_numbers().private_value.to_bytes(SCALAR_BYTES, "big")


def require_aware(moment: datetime, name: str) -> datetime:
    """Return ``moment`` unchanged, or raise ``ValueError`` when it is a naive ``datetime``."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return moment


class PrivateScalar:
    """A 32-byte P-256 private scalar. It never appears in ``repr``, ``str``, copies, or pickles."""

    __slots__ = ("_value",)

    def __init__(self, value: bytes) -> None:
        if type(value) is not bytes or len(value) != SCALAR_BYTES:
            raise ValueError(f"a private scalar is exactly {SCALAR_BYTES} bytes")
        self._value = value

    def key(self) -> ec.EllipticCurvePrivateKey:
        """Return the private key. ``ValueError`` if the scalar is not in ``[1, n-1]``."""
        return ec.derive_private_key(int.from_bytes(self._value, "big"), ec.SECP256R1())

    def __repr__(self) -> str:
        return "PrivateScalar(<redacted>)"

    __str__ = __repr__

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("a private key cannot be pickled")

    def __copy__(self) -> PrivateScalar:
        raise TypeError("a private key cannot be copied")

    def __deepcopy__(self, memo: object) -> PrivateScalar:
        raise TypeError("a private key cannot be copied")
