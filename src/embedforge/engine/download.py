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
import os
import shutil
import subprocess
import tempfile
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

EXPORTER_PACKAGE = "optimum-onnx[onnxruntime]"
"""Run through `uvx`, so exporting costs no project dependency.

The exporter needs PyTorch, which is far larger than everything this server uses put
together and is needed exactly once per model. Keeping it out of the project means the
runtime image stays small and CI does not install a deep learning framework to run
unit tests.
"""


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
    """Fetch every file for a model and write a manifest of their digests.

    Models that publish no ONNX build are exported locally instead; see
    `export_model`.
    """
    from huggingface_hub import hf_hub_download

    if config.needs_export:
        return export_model(settings, model_id, config, force=force)

    directory = model_directory(settings, model_id)
    if force and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for name in config.files:
        log.info("model_file_download", model=model_id, file=name, repo=config.repo_id)
        hf_hub_download(
            repo_id=config.repo_id,
            filename=name,
            revision=config.revision,
            local_dir=str(directory),
        )

    records = _record_files(directory, config.files)
    manifest = _write_manifest(settings, model_id, config, records, config.repo_id)
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


def _record_files(directory: Path, names: tuple[str, ...]) -> dict[str, FileRecord]:
    records: dict[str, FileRecord] = {}
    for name in names:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Expected {path} after fetching the model, but it is missing.")
        records[name] = FileRecord(sha256=sha256_file(path), size=path.stat().st_size)
    return records


def _write_manifest(
    settings: Settings,
    model_id: str,
    config: OnnxModelConfig,
    records: dict[str, FileRecord],
    source: str,
) -> Manifest:
    manifest = Manifest(
        model_id=model_id,
        repo_id=source,
        revision=config.revision,
        downloaded_at=datetime.now(UTC),
        files=records,
    )
    manifest_path(settings, model_id).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def export_model(
    settings: Settings,
    model_id: str,
    config: OnnxModelConfig,
    *,
    force: bool = False,
) -> Manifest:
    """Build an ONNX graph locally, for a model whose authors publish none.

    The source repository is fetched at its pinned revision first and exported from
    that local copy, so the result is reproducible in the same way a download is. We
    own the correctness of an export we produced, which is why `model verify` still
    records digests for it.
    """
    from huggingface_hub import snapshot_download

    assert config.export_from is not None
    if shutil.which("uvx") is None:
        raise RuntimeError(
            "Exporting needs `uvx` (part of uv: https://docs.astral.sh/uv/). "
            "Install uv, or run the export yourself:\n"
            f"  uvx --from '{EXPORTER_PACKAGE}' optimum-cli export onnx "
            f"--model {config.export_from} --task feature-extraction "
            f"{model_directory(settings, model_id)}"
        )

    directory = model_directory(settings, model_id)
    if force and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    log.info("model_export_started", model=model_id, source=config.export_from)
    with tempfile.TemporaryDirectory(prefix=f".{model_id}-source-", dir=directory) as staging:
        snapshot_download(
            repo_id=config.export_from,
            revision=config.revision,
            local_dir=staging,
        )
        environment = dict(os.environ)
        # Without this uv resolves the CUDA build of torch: about a gigabyte of GPU
        # wheels for an export that runs on CPU.
        environment.setdefault("UV_TORCH_BACKEND", "cpu")
        subprocess.run(  # noqa: S603
            [
                "uvx",
                "--from",
                EXPORTER_PACKAGE,
                "optimum-cli",
                "export",
                "onnx",
                "--model",
                staging,
                "--task",
                "feature-extraction",
                str(directory),
            ],
            check=True,
            env=environment,
        )

    records = _record_files(directory, config.files)
    manifest = _write_manifest(settings, model_id, config, records, config.export_from)
    log.info(
        "model_exported",
        model=model_id,
        files=len(records),
        bytes=sum(record.size for record in records.values()),
    )
    return manifest


def remove_model(settings: Settings, model_id: str) -> bool:
    """Delete a model's files. Returns False if there was nothing to delete."""
    directory = model_directory(settings, model_id)
    if not directory.exists():
        return False
    shutil.rmtree(directory)
    log.info("model_removed", model=model_id, path=str(directory))
    return True
