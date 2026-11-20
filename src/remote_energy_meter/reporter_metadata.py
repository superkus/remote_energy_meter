"""The only device-identifying metadata a reporter may carry, validated exactly as requirement 2.7 states.

Every ingress and downstream schema (enrollment, roster state, control snapshots and events, MQTT config, the Home
Assistant registry) uses the same constraints and **rejects rather than truncates or repairs** a bad value, so a value
that passes here is the same value everywhere. A future platform, architecture or version format needs an explicit
registry and schema update, not permissive acceptance.

* ``display_name``: NFC-normalized UTF-8 of 1 to 128 bytes; no leading or trailing whitespace; no control characters,
  line or paragraph separators, unassigned, private-use or surrogate code points, or format characters (which include
  the bidirectional controls). Interior spaces are allowed. "Unassigned" follows the Unicode tables of the running
  Python, so a code point added by a newer Unicode is refused until Python is updated.
* ``platform`` and ``architecture``: 1 to 32 bytes matching ``[a-z0-9][a-z0-9._-]{0,31}`` and, for now, a registered
  value: ``macos``; ``arm64`` or ``x86_64``.
* ``reporter_version``: 1 to 64 bytes of canonical ASCII Semantic Version 2.0.0 without build metadata.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from remote_energy_meter.protocol_v1 import JsonValue

__all__ = [
    "MAX_DISPLAY_NAME_BYTES",
    "MAX_VERSION_BYTES",
    "REGISTERED_ARCHITECTURES",
    "REGISTERED_PLATFORMS",
    "ReporterMetadata",
]

MAX_DISPLAY_NAME_BYTES: Final = 128
MAX_VERSION_BYTES: Final = 64
REGISTERED_PLATFORMS: Final = frozenset({"macos"})
REGISTERED_ARCHITECTURES: Final = frozenset({"arm64", "x86_64"})

_TOKEN: Final = re.compile(r"[a-z0-9][a-z0-9._-]{0,31}\Z")
# Semantic Versioning 2.0.0, section 2 grammar, without the build-metadata part.
_SEMVER: Final = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(-((0|[1-9][0-9]*|[0-9]*[a-zA-Z-][0-9a-zA-Z-]*)(\.(0|[1-9][0-9]*|[0-9]*[a-zA-Z-][0-9a-zA-Z-]*))*))?\Z"
)
# Control, format (including bidirectional controls), surrogate, private use, unassigned, line and paragraph separator.
_FORBIDDEN_CATEGORIES: Final = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})
_KEYS: Final = frozenset({"architecture", "display_name", "platform", "reporter_version"})


def _check_display_name(value: object) -> str:
    if type(value) is not str:
        raise ValueError("display_name must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("display_name contains a surrogate code point") from None
    if not 1 <= len(encoded) <= MAX_DISPLAY_NAME_BYTES:
        raise ValueError(f"display_name must be 1 to {MAX_DISPLAY_NAME_BYTES} UTF-8 bytes")
    if not unicodedata.is_normalized("NFC", value):
        raise ValueError("display_name must be NFC-normalized")
    if value != value.strip():
        raise ValueError("display_name must not start or end with whitespace")
    if any(unicodedata.category(character) in _FORBIDDEN_CATEGORIES for character in value):
        raise ValueError("display_name contains a control, format, separator, private-use or unassigned character")
    return value


def _check_token(value: object, name: str, registered: frozenset[str]) -> str:
    if type(value) is not str or not _TOKEN.fullmatch(value):
        raise ValueError(f"{name} must be 1 to 32 characters of [a-z0-9][a-z0-9._-]")
    if value not in registered:
        raise ValueError(f"{name} {value!r} is not a registered value")
    return value


def _check_version(value: object) -> str:
    if type(value) is not str or not value.isascii() or not 1 <= len(value) <= MAX_VERSION_BYTES:
        raise ValueError(f"reporter_version must be 1 to {MAX_VERSION_BYTES} ASCII bytes")
    if not _SEMVER.fullmatch(value):
        raise ValueError("reporter_version must be a Semantic Version 2.0.0 without build metadata")
    return value


@dataclass(frozen=True, slots=True)
class ReporterMetadata:
    """A reporter's display name, platform, architecture and version, validated on construction.

    Raises:
        ValueError: Any field violates requirement 2.7. Nothing is truncated or repaired.
    """

    display_name: str
    platform: str
    architecture: str
    reporter_version: str

    def __post_init__(self) -> None:
        _check_display_name(self.display_name)
        _check_token(self.platform, "platform", REGISTERED_PLATFORMS)
        _check_token(self.architecture, "architecture", REGISTERED_ARCHITECTURES)
        _check_version(self.reporter_version)

    def to_json(self) -> dict[str, JsonValue]:
        """Encode as plain JSON values."""
        return {
            "architecture": self.architecture,
            "display_name": self.display_name,
            "platform": self.platform,
            "reporter_version": self.reporter_version,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, JsonValue] | JsonValue) -> ReporterMetadata:
        """Decode exactly the four fields; ``ValueError`` for a missing, extra, or invalid one."""
        if not isinstance(raw, dict) or set(raw) != _KEYS:
            raise ValueError("metadata must have exactly display_name, platform, architecture and reporter_version")
        return cls(
            display_name=_as_text(raw["display_name"], "display_name"),
            platform=_as_text(raw["platform"], "platform"),
            architecture=_as_text(raw["architecture"], "architecture"),
            reporter_version=_as_text(raw["reporter_version"], "reporter_version"),
        )


def _as_text(value: JsonValue, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    return value
