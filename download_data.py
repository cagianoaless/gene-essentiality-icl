#!/usr/bin/env python3
"""Download Tahoe data and reconstruct the published split and context identities."""
import argparse
import itertools
import shutil
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from icl.data import h5column, read_counts, training_targets
from icl.io import ROOT, STRATA, BUDGETS, pair_sha, read, require, sha, write
from icl.selection import trajectories


def download(data_root):
    for record in read(ROOT / "data/sources.json")["inputs"]:
        destination = data_root / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(sha(destination) == record["sha256"], "Checksum mismatch: " + str(destination))
            continue
        temporary = destination.with_name(destination.name + ".part")
        temporary.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(record["url"], timeout=60) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            require(temporary.stat().st_size == record["bytes"], "File size mismatch: " + record["path"])
            require(sha(temporary) == record["sha256"], "SHA-256 mismatch: " + record["path"])
            temporary.replace(destination)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download " + record["path"] +
                ". Obtain the exact Tahoe file and place it at " + str(destination)
            ) from error


def _eligible_matrix(train, masks):
    genes = sorted(set().union(*(set(index.get_level_values(1)) for index in masks.values())))
    require(len(genes) == 2566, "Gene panel differs from the archived benchmark")
    row = {cell: i for i, cell in enumerate(train)}
    col = {gene: i for i, gene in enumerate(genes)}
    eligible = np.zeros((len(train), len(genes)), dtype=bool)
    for index in masks.values():
        for cell, gene in index:
            if cell in row:
                eligible[row[cell], col[gene]] = True
    return eligible


def _lung_trajectories(train, eligible):
    bits = [sum(1 << int(j) for j in np.flatnonzero(a)) for a in eligible]
    target = (1 << eligible.shape[1]) - 1
    valid = [(train[i], train[j]) for i, j in itertools.combinations(range(len(train)), 2)
             if bits[i] | bits[j] == target]
    require(len(valid) == 58 and all("ACH-000682" in pair for pair in valid),
            "Lung coverage panel differs from the archived experiment")
    rng = lambda namespace, *c: np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([20261027, namespace, *c])))
    chosen = rng(1).choice(len(valid), size=5, replace=False)
    orders = []
    for seed, index in enumerate(chosen):
        first = list(valid[int(index)])
        remaining = [cell for cell in train if cell not in first]
        orders.append(first + rng(2, seed).permutation(remaining).tolist())
    return len(valid), orders


def _validate_contexts(masks, selections):
    reference = read(ROOT / "expected/contexts.json")
    main = read(ROOT / "results/main/contexts.json")
    checked = 0
    for study, folds in (("depmap", range(5)), ("lung_ood", [0])):
        for fold in folds:
            if study == "depmap":
                frame = pd.read_csv(ROOT / "data/metadata/splits.csv")
                train = sorted(frame.loc[frame[f"fold-{fold}"] == "train", "cell-line"])
                test = sorted(frame.loc[frame[f"fold-{fold}"] == "val", "cell-line"])
            else:
                train = selections["lung_ood"]["train_cells"]
                test = selections["lung_ood"]["test_cells"]
            orders = [r["ordered_cells"] for r in selections[study]["trajectories"] if r["fold"] == fold]
            require(len(orders) == 5, "Expected five cell-selection trajectories")
            for stratum in STRATA:
                pair_index = masks[stratum]
                pool = pair_index[pair_index.get_level_values(0).isin(train)]
                query = pair_index[pair_index.get_level_values(0).isin(test)]
                if study == "depmap":
                    rng = np.random.default_rng(fold)
                else:
                    rng = np.random.Generator(
                        np.random.PCG64(np.random.SeedSequence([20261027, 6, STRATA.index(stratum)])))
                full = pool[rng.permutation(len(pool))]
                qhash = pair_sha(query)
                for budget in BUDGETS:
                    seeds = [None] if budget == "FULL" else range(5)
                    for seed in seeds:
                        selected = train if seed is None else orders[seed][:int(budget)]
                        context = full[full.get_level_values(0).isin(selected)]
                        prefix = f"fold{fold}/" if study == "depmap" else ""
                        tag = "shared" if seed is None else str(seed)
                        cid = f"{study}/{prefix}{stratum}/selection{tag}/C{budget}"
                        require(cid in reference, "Missing original context: " + cid)
                        expected = reference[cid]
                        require(pair_sha(context) == expected["context_sha256"]
                                and qhash == expected["query_sha256"],
                                "Reconstructed context differs from published benchmark: " + cid)
                        checked += 1
                if study == "depmap":
                    cid = f"F_C32768/fold{fold}/{stratum}"
                    expected = main[cid]
                    require(pair_sha(full[:32768]) == expected["context_sha256"]
                            and qhash == expected["query_sha256"],
                            "Main benchmark context differs: " + cid)
                    checked += 1
    return checked


def build_protocol(data_root):
    output = ROOT / "data/metadata"
    require(not output.exists(), "Protocol directory already exists: " + str(output))
    counts, cells, stable = read_counts(data_root / "counts.h5ad")
    require(len(cells) == 661 and len(stable) == 17172, "Unexpected counts dimensions")
    with h5py.File(data_root / "counts.h5ad", "r") as handle:
        symbols = h5column(handle["var"], "feature_name").astype(str)
    available = {}
    for index, symbol in enumerate(symbols):
        available.setdefault(symbol, index)

    stats = pd.read_csv(data_root / "misc/genes-by-mean-disc.csv")
    require({"gene", "null-frac", "mean-disc-dep"} <= set(stats.columns),
            "Invalid Tahoe gene-statistics table")
    masks, used = {}, set()
    for stratum in STRATA:
        low, high = (int(x) / 100 for x in stratum.split("to"))
        genes = sorted(stats.loc[
            (stats["null-frac"] < 0.1)
            & stats["mean-disc-dep"].between(low, high, inclusive="both")
            & stats["gene"].isin(available), "gene"
        ].astype(str).unique())
        used.update(genes)
        masks[stratum] = genes
    require(len(used) == 2566, "Expected 2,566 genes in the released nine strata")

    targets = training_targets(data_root / "raw/depmap-gene-dependencies.csv",
                               list(cells), sorted(used))
    output.mkdir(parents=True)
    pair_masks = {}
    for stratum, genes in masks.items():
        pairs = [(cell, gene) for gene in genes
                 for cell in targets.index[targets[gene].notna()]]
        pairs.sort()
        index = pd.MultiIndex.from_tuples(pairs, names=["cell-line", "gene"])
        require(index.is_unique, "Duplicate (cell-line, gene) pair")
        pair_masks[stratum] = index
        pd.DataFrame(pairs, columns=["cell-line", "gene"]).to_parquet(
            output / f"examples-{stratum}.parquet", index=False)

    split_source = data_root / "misc/split-cls.csv"
    split = pd.read_csv(split_source)
    require(set(cells) == set(split["cell-line"]), "Original Tahoe split differs from expression cells")
    shutil.copyfile(split_source, output / "splits.csv")

    mapping = pd.DataFrame([{
        "gene": gene, "expression_column": available[gene], "stable_id": stable[available[gene]]
    } for gene in sorted(used)])
    mapping.to_csv(output / "expression_mapping.csv", index=False)

    meta = pd.read_csv(data_root / "raw/depmap-metadata.csv")
    require("ModelID" in meta and "OncotreeLineage" in meta, "Lung metadata fields missing")
    lung = sorted(set(meta.loc[meta["OncotreeLineage"].eq("Lung"), "ModelID"]) & set(cells))
    non_lung = sorted(set(cells) - set(lung))
    require(len(lung) == 106 and len(non_lung) == 555, "Unexpected Lung cohort")

    selections = {
        "depmap": {"qualifying_C2_pairs": {}, "trajectories": []},
        "lung_ood": {"train_cells": non_lung, "test_cells": lung,
                     "qualifying_C2_pairs": {}, "trajectories": []}
    }
    for study, folds in (("depmap", range(5)), ("lung_ood", [0])):
        for fold in folds:
            if study == "depmap":
                train = sorted(split.loc[split[f"fold-{fold}"] == "train", "cell-line"])
                valid = _eligible_matrix(train, pair_masks)
                complete, trajectories_data = trajectories(train, valid, fold)
                count = len(complete)
                orders = [r["ordered_trajectory"] for r in trajectories_data]
            else:
                train = non_lung
                valid = _eligible_matrix(train, pair_masks)
                count, orders = _lung_trajectories(train, valid)
            selections[study]["qualifying_C2_pairs"][str(fold)] = count
            selections[study]["trajectories"].extend(
                {"fold": fold, "ordered_cells": order} for order in orders)

    write(output / "selections.json", selections)
    manifest_files = ["splits.csv", "expression_mapping.csv", "selections.json"] + [
        f"examples-{stratum}.parquet" for stratum in STRATA]
    write(output / "manifest.json", {filename: sha(output / filename) for filename in manifest_files})
    verified = _validate_contexts(pair_masks, selections)
    print(f"Reconstructed metadata: {verified} original context identities verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/tahoe"))
    parser.add_argument("--skip-download", action="store_true",
                        help="Validate files placed in --out and build protocol metadata")
    args = parser.parse_args()
    root = args.out.resolve()
    if not args.skip_download:
        download(root)
    else:
        for record in read(ROOT / "data/sources.json")["inputs"]:
            require(sha(root / record["path"]) == record["sha256"],
                    "Missing or changed Tahoe file: " + record["path"])
    build_protocol(root)


if __name__ == "__main__":
    main()
