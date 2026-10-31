"""Small atomic report writers and streaming file identities."""

import hashlib
import json
import re
import tempfile
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

FEATURE_ARTIFACTS = (
    "core_features.parquet",
    "core_feature_provenance.parquet",
    "core_labels.parquet",
    "core_feature_metadata.json",
)
MATCH_INPUTS = ("data/core_matches.parquet", "data/open_matches.parquet")


def source_paths(root: Path) -> tuple[str, ...]:
    return tuple(str(p.relative_to(root)) for p in sorted((root / "src/quant_football").glob("*.py"))) + (
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
    )


def contained_file(base: Path, relative: str) -> Path:
    """Reject non-canonical names and symlinks escaping the declared base."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Expected a canonical relative file path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative:
        raise ValueError(f"Unsafe relative file path: {relative}")
    base = base.resolve()
    result = (base / relative).resolve()
    if not result.is_relative_to(base) or result == base:
        raise ValueError(f"File escapes declared directory: {relative}")
    return result


def verify_hashes(base: Path, hashes: dict, expected_paths) -> None:
    if not isinstance(hashes, dict) or set(hashes) != set(expected_paths):
        raise ValueError("Hash inventory differs from the expected file set")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"Invalid SHA-256 identity: {name}")
        path = contained_file(base, name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"File changed: {name}; rebuild affected artifacts")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_parquet(path: Path, frame: pd.DataFrame, *, index: bool = False) -> None:
    """Write a fresh same-directory file, then atomically replace the final name.

    A pre-existing final-component symlink is replaced, never followed. A failed
    serialization leaves the previous artifact intact. This is not a transaction
    spanning several files; their report hashes are written only after success.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary, index=index)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class ArtifactStore:
    """Constrain generated paths; preflight a whole run before its first write.

    The explicitly selected base may resolve through a symlink. Descendant
    directory symlinks are rejected, including links that stay inside the base.
    Final-component links may be atomically replaced by the low-level writers.
    This guards accidental/pre-existing redirection, not hostile concurrent writers.
    """

    def __init__(self, base: Path, protected: Path | None = None):
        self.base = base.resolve()
        self.protected = None if protected is None else protected.resolve()
        if self.protected is not None and self.base.is_relative_to(self.protected):
            raise ValueError("Generated outputs must not be inside supplied data")

    def path(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("Expected a canonical output name")
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts or str(rel) != relative or relative == ".":
            raise ValueError(f"Unsafe output name: {relative}")
        path = self.base / relative
        current = self.base
        for part in rel.parts[:-1]:
            current /= part
            if current.is_symlink():
                raise ValueError(f"Output ancestor is a symlink: {current}")
            if current.exists() and not current.is_dir():
                raise ValueError(f"Output ancestor is not a directory: {current}")
        parent = path.parent.resolve()
        if not parent.is_relative_to(self.base):
            raise ValueError(f"Output escapes its declared directory: {relative}")
        if self.protected is not None and parent.is_relative_to(self.protected):
            raise ValueError(f"Output would replace supplied data: {relative}")
        if path.is_dir() and not path.is_symlink():
            raise ValueError(f"Output destination is a directory: {relative}")
        return parent / path.name

    def preflight(self, names) -> None:
        for name in names:
            self.path(name)

    def json(self, relative: str, value: Any) -> None:
        write_json(self.path(relative), value)

    def parquet(self, relative: str, frame: pd.DataFrame, *, index: bool = False) -> None:
        write_parquet(self.path(relative), frame, index=index)

    def text(self, relative: str, content: str) -> None:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            temporary.replace(self.path(relative))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def copy(self, relative: str, source: Path) -> None:
        import shutil

        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
            shutil.copyfile(source, temporary)
            temporary.replace(self.path(relative))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
