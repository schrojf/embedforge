"""Offline model comparison."""

import numpy as np
import pytest

from embedforge.comparison import (
    ComparisonResult,
    ModelRun,
    compare_models,
    cosine_similarity,
    spearman,
)
from embedforge.config import Settings


def make_run(model_id: str, documents: np.ndarray, queries: np.ndarray) -> ModelRun:
    return ModelRun(
        model_id=model_id,
        name=model_id,
        dimension=documents.shape[1],
        symmetric=True,
        load_seconds=0.1,
        embed_seconds=0.2,
        item_count=len(documents) + len(queries),
        document_vectors=documents,
        query_vectors=queries,
    )


def test_cosine_similarity_is_scale_invariant() -> None:
    left = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    right = np.array([[2.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    matrix = cosine_similarity(left, right)
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, [[1.0, -1.0], [0.0, 0.0]], atol=1e-6)


def test_cosine_similarity_survives_a_zero_vector() -> None:
    zero = np.zeros((1, 3), dtype=np.float32)
    assert np.isfinite(cosine_similarity(zero, zero)).all()


def test_spearman_detects_identical_and_reversed_orderings() -> None:
    values = np.array([3.0, 1.0, 2.0])
    assert spearman(values, values) == pytest.approx(1.0)
    assert spearman(values, -values) == pytest.approx(-1.0)
    assert np.isnan(spearman(np.array([1.0]), np.array([1.0])))
    # All-equal scores are one tied group, not an ordering, so correlation is undefined.
    assert np.isnan(spearman(values, np.ones(3)))
    assert np.isnan(spearman(np.ones(3), values))


def test_spearman_averages_ranks_across_ties() -> None:
    """Tied scores must not be given an ordering that argsort invented."""
    tied = np.array([5.0, 5.0, 1.0])
    assert spearman(tied, np.array([5.0, 5.0, 1.0])) == pytest.approx(1.0)
    assert spearman(tied, np.array([2.0, 2.0, 9.0])) == pytest.approx(-1.0)
    # Swapping the two tied entries changes nothing, because they are not ordered.
    assert spearman(tied, np.array([7.0, 3.0, 1.0])) == pytest.approx(
        spearman(tied, np.array([3.0, 7.0, 1.0]))
    )


def test_ranking_is_ordered_best_first() -> None:
    documents = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    run = make_run("m", documents, queries)

    ranking = run.ranking(0, top_k=3)
    assert [index for index, _ in ranking] == [0, 2, 1]
    assert [score for _, score in ranking] == sorted((score for _, score in ranking), reverse=True)
    assert run.ranking(0, top_k=2) == ranking[:2]


def test_agreement_is_one_for_identical_and_negative_for_reversed() -> None:
    documents = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    same = make_run("same", documents, queries)
    # Negating the query inverts every score, so the ranking is exactly reversed.
    reversed_run = make_run("reversed", documents, -queries)

    result = ComparisonResult(documents=["a", "b", "c"], queries=["q"], runs=[same, reversed_run])
    assert result.agreement(same, same) == pytest.approx(1.0)
    assert result.agreement(same, reversed_run) == pytest.approx(-1.0)


def test_agreement_is_undefined_without_queries() -> None:
    documents = np.eye(2, dtype=np.float32)
    run = make_run("m", documents, np.empty((0, 2), dtype=np.float32))
    result = ComparisonResult(documents=["a", "b"], queries=[], runs=[run])
    assert np.isnan(result.agreement(run, run))


def test_per_item_timing_is_reported() -> None:
    run = make_run("m", np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)[:1])
    assert run.item_count == 5
    assert run.ms_per_item == pytest.approx(0.2 / 5 * 1000)
    assert run.items_per_second == pytest.approx(25.0)


def test_compare_models_runs_each_model_over_the_same_inputs() -> None:
    result = compare_models(
        Settings(), ["dev-hash", "dev-hash"], ["first", "second", "third"], ["a query"]
    )
    assert [run.model_id for run in result.runs] == ["dev-hash", "dev-hash"]
    assert result.failures == {}
    for run in result.runs:
        assert run.document_vectors.shape == (3, 384)
        assert run.query_vectors.shape == (1, 384)
        assert run.item_count == 4
        assert run.embed_seconds > 0


def test_compare_models_works_without_queries() -> None:
    result = compare_models(Settings(), ["dev-hash"], ["first", "second"])
    run = result.runs[0]
    assert run.query_vectors.shape == (0, 384)
    assert run.document_similarity().shape == (2, 2)
    assert run.document_similarity()[0][0] == pytest.approx(1.0, abs=1e-5)


def test_a_symmetric_model_ranks_identically_from_embed_and_query() -> None:
    """dev-hash is symmetric, so a query equal to a document must match it exactly."""
    result = compare_models(Settings(), ["dev-hash"], ["alpha", "beta"], ["alpha"])
    best_index, best_score = result.runs[0].ranking(0, top_k=1)[0]
    assert best_index == 0
    assert best_score == pytest.approx(1.0, abs=1e-5)


def test_an_unloadable_model_is_reported_not_fatal() -> None:
    result = compare_models(Settings(), ["dev-hash", "no-such-model"], ["text"])
    assert [run.model_id for run in result.runs] == ["dev-hash"]
    assert "no-such-model" in result.failures


def test_rounds_average_several_timed_passes() -> None:
    result = compare_models(Settings(), ["dev-hash"], ["text"], rounds=3)
    assert result.runs[0].embed_seconds > 0


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="At least one document"):
        compare_models(Settings(), ["dev-hash"], [])
    with pytest.raises(ValueError, match="rounds"):
        compare_models(Settings(), ["dev-hash"], ["text"], rounds=0)
