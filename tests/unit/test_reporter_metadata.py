"""Reporter metadata: requirement 2.7, rejected rather than repaired, with every boundary it names."""

from __future__ import annotations

from typing import Any

import pytest

from remote_energy_meter import reporter_metadata
from remote_energy_meter.reporter_metadata import (
    MAX_DISPLAY_NAME_BYTES,
    MAX_VERSION_BYTES,
    ReporterMetadata,
)


def meta(**overrides: Any) -> ReporterMetadata:
    values: dict[str, Any] = {
        "display_name": "Studio Mac",
        "platform": "macos",
        "architecture": "arm64",
        "reporter_version": "1.2.3",
    }
    values.update(overrides)
    return ReporterMetadata(**values)


@pytest.mark.parametrize(
    "name",
    [
        "A",
        "Studio Mac",
        "Büro-Mac",  # precomposed (NFC) letters
        "名前",
        "Mac 🖥",
        "a" * MAX_DISPLAY_NAME_BYTES,
        "é" * (MAX_DISPLAY_NAME_BYTES // 2),  # two bytes each: exactly 128
        "with  interior   spaces",
    ],
)
def test_valid_display_names_are_accepted_unchanged(name: str) -> None:
    assert meta(display_name=name).display_name == name


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("", "1 to 128"),
        ("a" * (MAX_DISPLAY_NAME_BYTES + 1), "1 to 128"),
        ("é" * (MAX_DISPLAY_NAME_BYTES // 2 + 1), "1 to 128"),  # 130 bytes in 65 characters
        (" leading", "whitespace"),
        ("trailing ", "whitespace"),
        ("\u00a0nbsp", "whitespace"),
        ("ideographic\u3000", "whitespace"),
        ("e\u0301", "NFC"),  # decomposed: not normalized, and never repaired
        ("a\x00b", "control"),
        ("a\x1fb", "control"),
        ("a\x7fb", "control"),
        ("a\x85b", "control"),  # NEL
        ("a\tb", "control"),
        ("a\nb", "control"),
        ("a\u2028b", "separator"),  # line separator
        ("a\u2029b", "separator"),  # paragraph separator
        ("a\u200bb", "format"),  # zero-width space
        ("a\u202eb", "format"),  # right-to-left override (a bidirectional control)
        ("a\u2066b", "format"),  # left-to-right isolate
        ("a\ufeffb", "format"),  # byte order mark
        ("a\u00adb", "format"),  # soft hyphen
        ("a\ue000b", "private-use"),
        ("a\U000f0000b", "private-use"),
        ("a\u0378b", "unassigned"),
        ("a\U0010ffffb", "unassigned"),  # a noncharacter
        ("a\ud800b", "surrogate"),
    ],
)
def test_invalid_display_names_are_rejected_not_repaired(name: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        meta(display_name=name)


@pytest.mark.parametrize("value", [None, 5, b"bytes", ["a"], True])
def test_a_display_name_must_be_a_string(value: Any) -> None:
    with pytest.raises(ValueError, match="string"):
        meta(display_name=value)


def test_only_registered_platforms_and_architectures_are_accepted() -> None:
    assert meta(platform="macos", architecture="arm64").architecture == "arm64"
    assert meta(architecture="x86_64").architecture == "x86_64"
    for platform in ("linux", "windows", "MacOS", "mac os", "", "m" * 33):
        with pytest.raises(ValueError, match="platform"):
            meta(platform=platform)
    for architecture in ("aarch64", "amd64", "ARM64", "", "x86-64", "a" * 33):
        with pytest.raises(ValueError, match="architecture"):
            meta(architecture=architecture)


@pytest.mark.parametrize(
    "version",
    [
        "0.0.0",
        "1.2.3",
        "10.20.30",
        "1.2.3-rc.1",
        "1.2.3-alpha",
        "1.2.3-0.3.7",
        "1.2.3-x-y-z.--",
        "1.0.0-" + "a" * (MAX_VERSION_BYTES - 6),
    ],
)
def test_canonical_semantic_versions_without_build_metadata_are_accepted(version: str) -> None:
    assert meta(reporter_version=version).reporter_version == version


@pytest.mark.parametrize(
    "version",
    [
        "",
        "1",
        "1.2",
        "1.2.3.4",
        "v1.2.3",
        "01.2.3",
        "1.02.3",
        "1.2.03",
        "1.2.3+build",
        "1.2.3-rc.1+build.5",
        "1.2.3-",
        "1.2.3-01",
        "1.2.3-rc..1",
        "1.2.3 ",
        " 1.2.3",
        "1.2.3\n",
        "1.2.\uff13",  # a fullwidth digit
        "1.0.0-" + "a" * (MAX_VERSION_BYTES - 5),
    ],
)
def test_other_versions_are_rejected(version: str) -> None:
    with pytest.raises(ValueError, match="reporter_version"):
        meta(reporter_version=version)


def test_json_round_trips_exactly_four_fields() -> None:
    original = meta(display_name="Büro-Mac", architecture="x86_64", reporter_version="2.0.0-beta.1")

    assert ReporterMetadata.from_json(original.to_json()) == original
    assert set(original.to_json()) == {"architecture", "display_name", "platform", "reporter_version"}


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "x",
        [],
        {},
        {"display_name": "a", "platform": "macos", "architecture": "arm64"},
        {**meta().to_json(), "extra": "x"},
        {**meta().to_json(), "display_name": 5},
        {**meta().to_json(), "platform": None},
        {**meta().to_json(), "reporter_version": "1.2.3+b"},
    ],
)
def test_decoding_is_strict(raw: Any) -> None:
    with pytest.raises(ValueError):
        ReporterMetadata.from_json(raw)


def test_the_module_exports_its_public_names() -> None:
    for name in reporter_metadata.__all__:
        assert hasattr(reporter_metadata, name)
