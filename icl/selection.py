"""Target-free uniform coverage-constrained C2 sampling and nested cell sets."""
import itertools
import numpy as np

BUDGETS = [2, 4, 8, 16, 32, 64, 128, 256, "FULL"]
MASTER_SEED = 20261026


def qualifying_pairs(eligible):
    a = np.asarray(eligible)
    if a.dtype != bool or a.ndim != 2 or not all(a.shape):
        raise ValueError("Expected nonempty Boolean cell by unique-gene matrix")
    if a.all(axis=1).any():
        raise ValueError("Historical minimum-two invariant fails: a complete cell exists")
    bits = [int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little") for row in a]
    target = (1 << a.shape[1]) - 1
    valid = [(i, j) for i, j in itertools.combinations(range(len(a)), 2) if bits[i] | bits[j] == target]
    if len(valid) < 5:
        raise ValueError(f"Need five distinct qualifying pairs; found {len(valid)}")
    return np.asarray(valid, dtype=np.int32)


def trajectories(cells, eligible, fold, master_seed=MASTER_SEED, budgets=BUDGETS):
    cells = list(cells)
    if cells != sorted(set(cells)) or len(cells) != len(eligible):
        raise ValueError("Canonical unique cell order required")
    valid = qualifying_pairs(eligible)
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([master_seed, 1, fold])))
    chosen = generator.choice(len(valid), size=5, replace=False)
    records = []
    for replicate, chosen_index in enumerate(chosen):
        first = valid[chosen_index].tolist()
        remaining = np.setdiff1d(np.arange(len(cells)), first)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([master_seed, 2, fold, replicate])))
        order = first + rng.permutation(remaining).tolist()
        if sorted(order) != list(range(len(cells))):
            raise ValueError("Not a cell permutation")
        selections, previous = [], set()
        for budget in budgets:
            n = len(cells) if budget == "FULL" else budget
            if n > len(cells):
                raise ValueError("Budget exceeds training cohort; no capping")
            selected = order[:n] if budget != "FULL" else list(range(len(cells)))
            if not previous < set(selected) or not eligible[selected].any(axis=0).all():
                raise ValueError("Strict nestedness or aggregate full coverage failed")
            previous = set(selected)
            counts = eligible[selected].sum(axis=0)
            selections.append({"budget": budget, "cells": [cells[i] for i in selected],
                               "covered_genes": len(counts), "observations_per_unique_gene": counts.tolist()})
        records.append({"fold": fold, "replicate": replicate, "qualifying_pair_index": int(chosen_index),
                        "c2": [cells[i] for i in first], "ordered_trajectory": [cells[i] for i in order],
                        "remaining_rng_components": [master_seed, 2, fold, replicate], "selections": selections})
    return valid, records


def filtered_positions(full_cells, selected):
    selected = list(selected)
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate selected cell")
    positions = np.flatnonzero(np.isin(full_cells, selected))
    return positions


def verify_context(full_cells, full_genes, positions, selected, train, held, gene_count):
    positions = np.asarray(positions)
    expected = filtered_positions(full_cells, selected)
    if not np.array_equal(positions, expected):
        raise ValueError("Context omitted/duplicated/reordered an eligible selected pair")
    if not set(selected) <= set(train) or set(selected) & set(held):
        raise ValueError("Held-out or unknown selected cell")
    counts = np.bincount(np.asarray(full_genes)[positions], minlength=gene_count)
    if len(counts) != gene_count or (counts == 0).any():
        raise ValueError("Incomplete per-gene training coverage; fallback forbidden")
    return counts
