"""Asset-free regression tests for the released preprocessing and evaluation code."""

import numpy as np
import pandas as pd
import pytest

from icl.data import feature_matrix
from icl.evaluation import aggregate, metrics
from icl.io import STRATA
from icl.preprocessing import ExpressionPCA
from icl.selection import filtered_positions, trajectories, verify_context


def test_pca_uses_training_expression_only():
    rng = np.random.default_rng(13)
    counts = rng.poisson(12, size=(24, 20)).astype(np.float32) + 1
    train = [f"CL{i:02d}" for i in range(20)]
    held = [f"CL{i:02d}" for i in range(20, 24)]
    pca = ExpressionPCA(15, 0).fit(counts[:20], train)
    pca.assert_training_only(train, held)
    assert pca.target_sum_ == np.median(counts[:20].sum(axis=1, dtype=np.float64))
    assert pca.transform(counts[20:]).shape == (4, 15)
    assert pca.gene_loadings().shape == (20, 15)
    with pytest.raises(ValueError):
        pca.assert_training_only(train, [train[0]])
    with pytest.raises(ValueError):
        pca.assert_training_only(train[1:], held)


def test_pair_features_are_15_plus_15_plus_1():
    cells = pd.Index(["CL0", "CL1", "CL2"])
    pairs = pd.MultiIndex.from_tuples([("CL2", "G3"), ("CL0", "G1")])
    scores = np.arange(45, dtype=np.float32).reshape(3, 15)
    loadings = np.arange(60, dtype=np.float32).reshape(4, 15) / 10
    expression = np.arange(12, dtype=np.float32).reshape(3, 4) / 100
    x = feature_matrix(pairs, cells, scores, expression, loadings,
                       {f"G{i}": i for i in range(4)}, "F")
    assert x.shape == (2, 31) and x.dtype == np.float32
    np.testing.assert_array_equal(x[0, :15], scores[2])
    np.testing.assert_array_equal(x[0, 15:30], loadings[3])
    assert x[0, 30] == expression[2, 3]
    np.testing.assert_array_equal(x[1, :15], scores[0])
    np.testing.assert_array_equal(x[1, 15:30], loadings[1])
    assert x[1, 30] == expression[0, 1]


def test_context_rejects_reordering_and_heldout_cells():
    all_cells = np.array(["A", "A", "B", "B", "C", "C"])
    all_genes = np.array([0, 1, 0, 1, 0, 1])
    selected = ["A", "B"]
    positions = filtered_positions(all_cells, selected)
    np.testing.assert_array_equal(
        verify_context(all_cells, all_genes, positions, selected,
                       ["A", "B"], ["C"], 2), [2, 2])
    with pytest.raises(ValueError):
        verify_context(all_cells, all_genes, positions[::-1], selected,
                       ["A", "B"], ["C"], 2)
    with pytest.raises(ValueError):
        verify_context(all_cells, all_genes, filtered_positions(all_cells, ["A", "C"]),
                       ["A", "C"], ["A", "B"], ["C"], 2)


def test_coverage_constrained_cell_selections_are_nested_and_repeatable():
    cells = [f"CL{i}" for i in range(8)]
    patterns = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool)
    eligible = patterns[[0, 1, 2, 0, 1, 2, 0, 1]]
    valid, first = trajectories(cells, eligible, 0, budgets=[2, 4, "FULL"])
    again_valid, again = trajectories(cells, eligible, 0, budgets=[2, 4, "FULL"])
    np.testing.assert_array_equal(valid, again_valid)
    assert first == again and len(first) == 5
    for trial in first:
        a, b, full = [set(x["cells"]) for x in trial["selections"]]
        assert len(a) == 2 and len(b) == 4 and len(full) == 8
        assert a < b < full
        assert eligible[[cells.index(c) for c in a]].any(axis=0).all()


def test_binary_metrics_use_dependency_threshold_half():
    y = np.array([0.2, 0.8, 0.4, 0.9], dtype=float)
    p = np.array([0.1, 0.75, 0.5, 0.85], dtype=float)
    scores = metrics(y, p)
    assert scores["auroc"] == 1.0
    assert scores["auprc"] == 1.0
    assert np.isfinite(scores["r2"])
    with pytest.raises(ValueError):
        metrics(y, p[:2])


def test_depmap_aggregation_uses_fold_mean_then_nine_stratum_median():
    rows = [dict(experiment="depmap", model="rf", budget="FULL", seed=None,
                 fold=fold, stratum=stratum, auroc=0.60 + i * 0.005 + fold * 0.001,
                 r2=0.03 + i * 0.01 + fold * 0.001,
                 auprc=0.50 + i * 0.01 + fold * 0.001)
            for i, stratum in enumerate(STRATA) for fold in range(5)]
    frame = pd.DataFrame(rows)
    endpoints, strata = aggregate(frame)
    assert len(endpoints) == 1 and len(strata) == 9
    assert bool(endpoints.iloc[0].complete)
    assert endpoints.iloc[0].auroc == pytest.approx(0.622)
    assert endpoints.iloc[0].r2 == pytest.approx(0.072)
    assert endpoints.iloc[0].auprc == pytest.approx(0.542)
    shuffled, _ = aggregate(frame.sample(frac=1, random_state=19))
    assert shuffled.iloc[0].auroc == pytest.approx(endpoints.iloc[0].auroc)
    with pytest.raises(ValueError):
        aggregate(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    incomplete, _ = aggregate(frame.iloc[1:])
    assert not bool(incomplete.iloc[0].complete)
    assert pd.isna(incomplete.iloc[0].auroc)
