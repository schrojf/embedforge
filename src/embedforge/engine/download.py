# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# huggingface_hub ships without complete type stubs.
"""Fetching and verifying model files.

Downloads happen here, never at server startup: a container should start fast, start
offline, and never half-download a model while traffic is arriving.

Every download records a manifest of sha256 digests, so `model verify` can tell a
complete model from a truncated one. Revisions are pinned commit shas, so the bytes
you verified today are the bytes you get tomorrow.
"""

import hashlib
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from embedforge.config import Settings
from embedforge.engine.onnx_backend import OnnxModelConfig, model_directory
from embedforge.logging import get_logger

log = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
_CHUNK = 1024 * 1024


class FileRecord(BaseModel):
    sha256: str
    size: int


class Manifest(BaseModel):
    """What was downloaded, from where, and what it hashed to."""

    version: int = 1
    model_id: str
    repo_id: str
    revision: str
    downloaded_at: datetime
    files: dict[str, FileRecord]


class FileStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    CORRUPT = "corrupt"
    UNVERIFIED = "unverified"
    """Present, but there is no manifest entry to check it against."""


@dataclass(frozen=True, slots=True)
class FileCheck:
    name: str
    status: FileStatus
    detail: str = ""


@dataclass(frozen=True, slots=True)
class VerificationReport:
    model_id: str
    directory: Path
    checks: list[FileCheck]
    has_manifest: bool

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.status is FileStatus.OK for check in self.checks)

    @property
    def problems(self) -> list[FileCheck]:
        return [check for check in self.checks if check.status is not FileStatus.OK]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(settings: Settings, model_id: str) -> Path:
    return model_directory(settings, model_id) / MANIFEST_NAME


def read_manifest(settings: Settings, model_id: str) -> Manifest | None:
    path = manifest_path(settings, model_id)
    if not path.exists():
        return None
    try:
        return Manifest.model_validate_json(path.read_text("utf-8"))
    except Exception as error:
        log.warning("manifest_invalid", model=model_id, error=str(error))
        return None


def is_downloaded(settings: Settings, model_id: str, config: OnnxModelConfig) -> bool:
    """Cheap presence check: are all the expected files on disk?"""
    directory = model_directory(settings, model_id)
    return all((directory / name).exists() for name in config.files)


def download_model(
    settings: Settings,
    model_id: str,
    config: OnnxModelConfig,
    *,
    force: bool = False,
) -> Manifest:
    """Fetch every file for a model and write a manifest of their digests."""
    from huggingface_hub import hf_hub_download

    directory = model_directory(settings, model_id)
    if force and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    records: dict[str, FileRecord] = {}
    for name in config.files:
        log.info("model_file_download", model=model_id, file=name, repo=config.repo_id)
        path = Path(
            hf_hub_download(
                repo_id=config.repo_id,
                filename=name,
                revision=config.revision,
                local_dir=str(directory),
            )
        )
        records[name] = FileRecord(sha256=sha256_file(path), size=path.stat().st_size)

    manifest = Manifest(
        model_id=model_id,
        repo_id=config.repo_id,
        revision=config.revision,
        downloaded_at=datetime.now(UTC),
        files=records,
    )
    manifest_path(settings, model_id).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    log.info(
        "model_downloaded",
        model=model_id,
        files=len(records),
        bytes=sum(record.size for record in records.values()),
    )
    return manifest


def verify_model(
    settings: Settings,
    model_id: str,
    config: OnnxModelConfig,
    *,
    deep: bool = True,
) -> VerificationReport:
    """Check the files on disk against the manifest recorded at download time.

    `deep=False` checks presence and size only, which is instant; the default also
    re-hashes, which is what catches a truncated or corrupted download.
    """
    directory = model_directory(settings, model_id)
    manifest = read_manifest(settings, model_id)
    checks: list[FileCheck] = []

    for name in config.files:
        path = directory / name
        if not path.exists():
            checks.append(FileCheck(name, FileStatus.MISSING, "not on disk"))
            continue
        record = manifest.files.get(name) if manifest else None
        if record is None:
            checks.append(FileCheck(name, FileStatus.UNVERIFIED, "no manifest entry"))
            continue
        size = path.stat().st_size
        if size != record.size:
            checks.append(
                FileCheck(
                    name,
                    FileStatus.CORRUPT,
                    f"expected {record.size} bytes, found {size}",
                )
            )
            continue
        if deep and sha256_file(path) != record.sha256:
            checks.append(FileCheck(name, FileStatus.CORRUPT, "sha256 mismatch"))
            continue
        checks.append(FileCheck(name, FileStatus.OK))

    return VerificationReport(
        model_id=model_id,
        directory=directory,
        checks=checks,
        has_manifest=manifest is not None,
    )


def remove_model(settings: Settings, model_id: str) -> bool:
    """Delete a model's files. Returns False if there was nothing to delete."""
    directory = model_directory(settings, model_id)
    if not directory.exists():
        return False
    shutil.rmtree(directory)
    log.info("model_removed", model=model_id, path=str(directory))
    return True
