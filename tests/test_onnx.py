"""ONNX backend, model catalog, and file management.

These tests never touch the network. The one test that needs real model files skips
itself when they are not present.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from embedforge.config import Settings
from embedforge.engine import registry
from embedforge.engine.base import Modality, TaskType, TextInput
from embedforge.engine.catalog import CATALOG
from embedforge.engine.download import (
    FileRecord,
    FileStatus,
    Manifest,
    is_downloaded,
    read_manifest,
    remove_model,
    sha256_file,
    verify_model,
)
from embedforge.engine.onnx_backend import (
    OnnxModelConfig,
    OnnxTextBackend,
    Pooling,
    model_directory,
    pool,
)

SHA_LENGTH = 40


# ---- Catalog invariants ----


def test_every_catalog_model_is_registered() -> None:
    known = set(registry.known_ids())
    for info, _ in CATALOG:
        assert info.id in known
        assert registry.is_onnx_model(info.id)
    assert not registry.is_onnx_model("dev-hash"), "the built-in model has no files"


def test_revisions_are_pinned_to_commit_shas() -> None:
    """A branch name would make downloads unreproducible and unverifiable."""
    for info, config in CATALOG:
        assert len(config.revision) == SHA_LENGTH, info.id
        assert all(character in "0123456789abcdef" for character in config.revision), info.id


def test_every_model_documents_its_trade_offs() -> None:
    for info, _ in CATALOG:
        assert info.pros, info.id
        assert info.cons, info.id
        assert info.description, info.id
        assert info.size_mb, info.id
        assert info.license != "unknown", info.id
        assert info.modalities == (Modality.TEXT,), info.id


def test_task_handling_matches_the_symmetry_flag() -> None:
    """A symmetric model must treat both tasks alike; an asymmetric one must not.

    Models do this by different mechanisms - a text prefix, or a task adapter selected
    inside the graph - so the invariant is about the effect, not the mechanism. Getting
    it wrong produces vectors that look fine and retrieve badly, which is worth failing
    the build over.
    """
    for info, config in CATALOG:
        assert config.distinguishes_tasks is not info.symmetric, info.id


def test_prefix_and_adapter_are_not_both_used() -> None:
    """A model selects its task one way or the other.

    jina-embeddings-v3's LoRA adapter replaces the instruction text entirely; sending a
    prefix as well would corrupt the input without any error.
    """
    for info, config in CATALOG:
        if config.query_task_id is not None:
            assert config.query_prefix == "", info.id
            assert config.document_prefix == "", info.id


def test_task_ids_are_declared_in_pairs() -> None:
    for info, config in CATALOG:
        assert (config.query_task_id is None) == (config.document_task_id is None), info.id


def test_exported_models_declare_a_source_and_a_flat_layout() -> None:
    """optimum writes its export flat, unlike a published build's nested onnx/ path."""
    for info, config in CATALOG:
        if not config.needs_export:
            continue
        assert config.export_from, info.id
        assert "/" not in config.onnx_file, info.id
        assert "/" not in config.tokenizer_file, info.id
        assert not config.extra_files, info.id


def test_context_length_matches_the_tokenizer_limit() -> None:
    for info, config in CATALOG:
        assert info.max_input_tokens == config.max_seq_length, info.id


def test_catalog_covers_both_short_and_long_context() -> None:
    """The catalog exists to be compared: it must span the real choices."""
    contexts = {info.max_input_tokens for info, _ in CATALOG}
    dimensions = {info.dimension for info, _ in CATALOG}
    assert max(contexts) >= 8192 and min(contexts) <= 512
    assert {384, 768, 1024} <= dimensions
    assert any(info.symmetric for info, _ in CATALOG)
    assert any(not info.symmetric for info, _ in CATALOG)


def test_onnx_config_lookup_rejects_non_file_models() -> None:
    from embedforge.errors import InvalidRequestError

    assert registry.onnx_config("e5-small").repo_id
    with pytest.raises(InvalidRequestError, match="no files to manage"):
        registry.onnx_config("dev-hash")
    with pytest.raises(InvalidRequestError, match="Unknown model"):
        registry.onnx_config("not-a-model")


# ---- Pooling ----


def test_mean_pooling_ignores_padding() -> None:
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)
    assert np.allclose(pool(hidden, mask, Pooling.MEAN), [[2.0, 3.0]])


def test_cls_pooling_takes_the_first_token() -> None:
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    mask = np.array([[1, 1]], dtype=np.int64)
    assert np.allclose(pool(hidden, mask, Pooling.CLS), [[1.0, 2.0]])


def test_mean_pooling_survives_an_all_padding_row() -> None:
    hidden = np.ones((1, 3, 2), dtype=np.float32)
    mask = np.zeros((1, 3), dtype=np.int64)
    assert np.isfinite(pool(hidden, mask, Pooling.MEAN)).all()


# ---- File layout ----


def test_files_mirror_the_repository_layout() -> None:
    config = registry.onnx_config("e5-large-instruct")
    assert "onnx/model.onnx" in config.files
    # External weights must stay beside the graph, so the subdirectory is preserved.
    assert "onnx/model.onnx_data" in config.files
    assert "tokenizer.json" in config.files


def test_variants_of_one_repository_get_separate_directories(tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path)
    base = model_directory(settings, "e5-base")
    quantized = model_directory(settings, "e5-base-int8")
    assert base != quantized
    assert registry.onnx_config("e5-base").repo_id == registry.onnx_config("e5-base-int8").repo_id


def test_loading_without_files_says_how_to_fix_it(tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, model_id="e5-small")
    backend = registry.create_backend(settings)
    with pytest.raises(FileNotFoundError, match="embedforge model download e5-small"):
        backend.load()


# ---- Download manifest and verification ----


def write_fake_model(tmp_path: Path, model_id: str, config: OnnxModelConfig) -> Settings:
    """Lay out plausible files plus a matching manifest, without any network."""
    settings = Settings(model_dir=tmp_path)
    directory = model_directory(settings, model_id)
    files: dict[str, FileRecord] = {}
    for name in config.files:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode() * 10)
        files[name] = FileRecord(sha256=sha256_file(path), size=path.stat().st_size)
    manifest = Manifest(
        model_id=model_id,
        repo_id=config.repo_id,
        revision=config.revision,
        downloaded_at=datetime(2026, 1, 1, tzinfo=UTC),
        files=files,
    )
    (directory / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    return settings


def test_verify_passes_for_an_intact_download(tmp_path: Path) -> None:
    config = registry.onnx_config("e5-small")
    settings = write_fake_model(tmp_path, "e5-small", config)
    assert is_downloaded(settings, "e5-small", config)

    report = verify_model(settings, "e5-small", config)
    assert report.ok
    assert report.has_manifest
    assert {check.status for check in report.checks} == {FileStatus.OK}


def test_verify_detects_a_truncated_file(tmp_path: Path) -> None:
    """The failure mode this exists for: a download that stopped early."""
    config = registry.onnx_config("e5-small")
    settings = write_fake_model(tmp_path, "e5-small", config)
    (model_directory(settings, "e5-small") / config.onnx_file).write_bytes(b"short")

    report = verify_model(settings, "e5-small", config)
    assert not report.ok
    assert report.problems[0].status is FileStatus.CORRUPT
    assert "bytes" in report.problems[0].detail


def test_verify_detects_silent_corruption_only_when_hashing(tmp_path: Path) -> None:
    config = registry.onnx_config("e5-small")
    settings = write_fake_model(tmp_path, "e5-small", config)
    path = model_directory(settings, "e5-small") / config.onnx_file
    original = path.read_bytes()
    # Same length, different bytes: size checks cannot see this.
    path.write_bytes(b"x" * len(original))

    assert verify_model(settings, "e5-small", config, deep=False).ok
    deep = verify_model(settings, "e5-small", config, deep=True)
    assert not deep.ok
    assert deep.problems[0].detail == "sha256 mismatch"


def test_verify_reports_missing_files(tmp_path: Path) -> None:
    config = registry.onnx_config("e5-small")
    settings = write_fake_model(tmp_path, "e5-small", config)
    (model_directory(settings, "e5-small") / config.onnx_file).unlink()

    report = verify_model(settings, "e5-small", config)
    assert not report.ok
    assert report.problems[0].status is FileStatus.MISSING
    assert not is_downloaded(settings, "e5-small", config)


def test_files_without_a_manifest_are_unverified_not_ok(tmp_path: Path) -> None:
    config = registry.onnx_config("e5-small")
    settings = write_fake_model(tmp_path, "e5-small", config)
    (model_directory(settings, "e5-small") / "manifest.json").unlink()

    report = verify_model(settings, "e5-small", config)
    assert not report.has_manifest
    assert {check.status for check in report.checks} == {FileStatus.UNVERIFIED}


def test_a_corrupt_manifest_is_not_fatal(tmp_path: Path) -> None:
    config = registry.onnx_config("e5-small")
    settings = write_fake_model(tmp_path, "e5-small", config)
    (model_directory(settings, "e5-small") / "manifest.json").write_text("{oops", encoding="utf-8")

    assert read_manifest(settings, "e5-small") is None
    assert not verify_model(settings, "e5-small", config).ok


def test_manifest_records_where_the_files_came_from(tmp_path: Path) -> None:
    config = registry.onnx_config("e5-small")
    settings = write_fake_model(tmp_path, "e5-small", config)
    manifest = read_manifest(settings, "e5-small")
    assert isinstance(manifest, Manifest)
    assert manifest.repo_id == config.repo_id
    assert manifest.revision == config.revision


def test_remove_deletes_the_files(tmp_path: Path) -> None:
    config = registry.onnx_config("e5-small")
    settings = write_fake_model(tmp_path, "e5-small", config)
    assert remove_model(settings, "e5-small") is True
    assert not model_directory(settings, "e5-small").exists()
    assert remove_model(settings, "e5-small") is False


# ---- Integration, only when the model files are actually present ----

INTEGRATION_MODEL = "e5-small-int8"


def integration_settings() -> Settings | None:
    settings = Settings(model_dir=Path("data/models"), model_id=INTEGRATION_MODEL)
    config = registry.onnx_config(INTEGRATION_MODEL)
    return settings if is_downloaded(settings, INTEGRATION_MODEL, config) else None


needs_model = pytest.mark.skipif(
    integration_settings() is None,
    reason=f"run 'embedforge model download {INTEGRATION_MODEL}' to enable",
)


@needs_model
def test_a_real_model_embeds_and_distinguishes_query_from_document() -> None:
    settings = integration_settings()
    assert settings is not None
    backend = registry.create_backend(settings)
    backend.load()
    try:
        assert isinstance(backend, OnnxTextBackend)
        info = backend.info

        documents = ["Mačka spí na gauči.", "The stock market fell sharply today."]
        vectors = backend.embed([TextInput(text) for text in documents], TaskType.DOCUMENT)
        assert vectors.shape == (2, info.dimension)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

        # Asymmetric model: the prefixes must actually change the output.
        as_query = backend.embed([TextInput(documents[0])], TaskType.QUERY)
        assert not np.allclose(as_query[0], vectors[0])

        # Real semantics: a Slovak query about the cat must prefer the cat sentence.
        query = backend.embed([TextInput("kde spí mačka?")], TaskType.QUERY)
        scores = query @ vectors.T
        assert scores[0][0] > scores[0][1]
    finally:
        backend.close()


# ---- Task selection ----


@dataclass
class StubNode:
    name: str


class StubSession:
    """Minimal stand-in for an ONNX session, to check what gets fed to it."""

    def __init__(self, input_names: list[str], dimension: int = 4) -> None:
        self._inputs = [StubNode(name) for name in input_names]
        self.dimension = dimension
        self.last_feeds: dict[str, np.ndarray] = {}

    def get_inputs(self) -> list[StubNode]:
        return self._inputs

    def run(self, _outputs: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.last_feeds = feeds
        batch, tokens = feeds["input_ids"].shape
        return [np.ones((batch, tokens, self.dimension), dtype=np.float32)]


class StubTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode_batch(self, texts: list[str]) -> list[object]:
        self.texts = texts
        return [
            type("Encoding", (), {"ids": [1, 2, 3], "attention_mask": [1, 1, 1]})() for _ in texts
        ]


def stub_backend(config: OnnxModelConfig, input_names: list[str]) -> OnnxTextBackend:
    from embedforge.engine.base import ModelInfo

    info = ModelInfo(id="stub", name="stub", dimension=4, max_input_tokens=8, symmetric=False)
    backend = OnnxTextBackend(info, config, Settings())
    backend._session = StubSession(input_names)  # pyright: ignore[reportPrivateUsage]
    backend._tokenizer = StubTokenizer()  # pyright: ignore[reportPrivateUsage]
    backend._input_names = frozenset(input_names)  # pyright: ignore[reportPrivateUsage]
    return backend


def test_task_id_is_fed_as_a_scalar_when_the_graph_wants_one() -> None:
    backend = stub_backend(
        registry.onnx_config("jina-v3"), ["input_ids", "attention_mask", "task_id"]
    )
    backend.embed([TextInput("hello")], TaskType.QUERY)
    session = backend._session  # pyright: ignore[reportPrivateUsage]
    assert session.last_feeds["task_id"] == 0
    # A 0-d array, as the model card's example uses: one adapter for the whole batch.
    assert session.last_feeds["task_id"].shape == ()

    backend.embed([TextInput("hello")], TaskType.DOCUMENT)
    assert session.last_feeds["task_id"] == 1


def test_undeclared_inputs_are_never_fed() -> None:
    """Feeding an input the graph does not declare is an error, not a no-op."""
    backend = stub_backend(registry.onnx_config("jina-v3"), ["input_ids", "attention_mask"])
    backend.embed([TextInput("hello")], TaskType.QUERY)
    session = backend._session  # pyright: ignore[reportPrivateUsage]
    assert set(session.last_feeds) == {"input_ids", "attention_mask"}


def test_prefixes_are_applied_per_task() -> None:
    backend = stub_backend(registry.onnx_config("e5-sk-large"), ["input_ids", "attention_mask"])
    backend.embed([TextInput("ahoj")], TaskType.QUERY)
    assert backend._tokenizer.texts == ["query: ahoj"]  # pyright: ignore[reportPrivateUsage]
    backend.embed([TextInput("ahoj")], TaskType.DOCUMENT)
    assert backend._tokenizer.texts == ["passage: ahoj"]  # pyright: ignore[reportPrivateUsage]
