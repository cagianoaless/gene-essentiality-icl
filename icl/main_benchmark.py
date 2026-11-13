"""Prepare, run and evaluate the principal DepMap benchmark."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import yaml

from .io import ROOT, STRATA, require, read, write, sha, array_sha, source_array_sha, pair_sha, save, load
from .data import read_counts, training_targets, targets_at, masks, cohorts, expression_state, feature_matrix
from .evaluation import metrics
from .models import make_model, model_execution, predict_model

BASE = ROOT / "results/main"


def build_unit(data_root, model, fold, stratum, context=None, pca_mode="frozen"):
    """Prepare one fold-stratum unit with training-only labels."""
    config = yaml.safe_load((ROOT / "configs/main_benchmark.yaml").read_text())
    require(model in config["models"] and fold in range(5) and stratum in STRATA,
            "Unregistered main unit")
    setting = config["models"][model]
    budget = setting["context"] if context is None else context
    require(budget == setting["context"] or (model == "tabpfn" and budget == 32768),
            "Context differs from recovered historical setting")
    data_root = Path(data_root).resolve()
    for item in read(ROOT / "data/sources.json")["inputs"]:
        require(sha(data_root / item["path"]) == item["sha256"], "Wrong raw snapshot")
    counts, cells, stable = read_counts(data_root / "counts.h5ad")
    mapping_frame = pd.read_csv(ROOT / "data/metadata/expression_mapping.csv")
    require(np.array_equal(stable[mapping_frame.expression_column], mapping_frame.stable_id),
            "Stable expression identity mapping differs")
    mapping = dict(zip(mapping_frame.gene, mapping_frame.expression_column))
    train, test = cohorts("depmap", fold)
    require(len(train) in [528, 529] and not set(train) & set(test)
            and set(train) | set(test) == set(cells), "Wrong released cell fold")
    scores, norm, loading, preprocessing = expression_state(
        counts, cells, train, test, f"depmap/fold{fold}", mode=pca_mode)
    pairs = masks()[stratum]
    pool = pairs[pairs.get_level_values(0).isin(train)]
    query = pairs[pairs.get_level_values(0).isin(test)]
    labels = training_targets(data_root / "raw/depmap-gene-dependencies.csv",
                              train, sorted(mapping), parser="pandas")
    full = pool[np.random.default_rng(fold).permutation(len(pool))]
    context_pairs = full if budget == "FULL" else full[:32768]
    require(not set(context_pairs.get_level_values(0)) & set(query.get_level_values(0)),
            "Query labels entered context")
    cid = f"F_{'FULL' if budget == 'FULL' else 'C32768'}/fold{fold}/{stratum}"
    frozen = read(BASE / "contexts.json")[cid]
    require(pair_sha(context_pairs) == frozen["context_sha256"]
            and pair_sha(query) == frozen["query_sha256"], "Changed context/query identities")
    X = feature_matrix(context_pairs, cells, scores, norm, loading, mapping, "F")
    Q = feature_matrix(query, cells, scores, norm, loading, mapping, "F")
    y = targets_at(labels, context_pairs)
    require(X.shape == (frozen["n_train"], 31)
            and Q.shape == (frozen["n_query"], 31), "Wrong F feature interface")
    refit_without_reference = preprocessing.get("reference_available") is False
    blocks = [("y", y)] if refit_without_reference else [
        ("X", X), ("Q", Q), ("y", y)]
    for name, values in blocks:
        require(source_array_sha(values, "depmap") == frozen[name + "_sha256"],
                "Numerical inputs differ from historical receipt: " + name)
    preprocessing["original_feature_hash_verified"] = not refit_without_reference
    return config, frozen, X, Q, y, context_pairs, query, preprocessing


def prepare(data_root, model, fold, stratum, out, context=None, pca_mode="frozen"):
    out = Path(out).resolve()
    require(not out.exists(), "Never overwrite prepared inputs")
    t = time.perf_counter()
    config, frozen, X, Q, y, pairs, query, preprocessing = build_unit(
        data_root, model, fold, stratum, context, pca_mode)
    out.mkdir(parents=True)
    pd.DataFrame(query.tolist(), columns=["ModelID", "gene"]).to_csv(out / "query_ids.csv", index=False)
    pd.DataFrame(pairs.tolist(), columns=["ModelID", "gene"]).to_csv(out / "context_ids.csv", index=False)
    manifest = dict(model=model, config=config, historical=frozen,
                    arrays={k:save(out / (k + ".npy"), a) for k,a in [("X",X),("Q",Q),("y",y)]},
                    query_identity_sha256=sha(out / "query_ids.csv"),
                    context_identity_sha256=sha(out / "context_ids.csv"),
                    preprocessing=preprocessing, query_truth_accessed=False,
                    preparation_seconds=time.perf_counter()-t)
    write(out / "manifest.json", manifest)
    return out / "manifest.json"


def implementation_hashes():
    return {name:sha(ROOT / name) for name in [
        "icl/main_benchmark.py", "icl/models.py", "icl/data.py", "icl/preprocessing.py",
        "icl/io.py", "configs/main_benchmark.yaml"]}


def run_unit(prepared, checkpoint, out):
    prepared, out = Path(prepared).resolve(), Path(out).resolve()
    require(not out.exists(), "Never overwrite/retry an inference attempt")
    m = read(prepared); model_name = m["model"]; c = m["historical"]
    require(m["config"] == yaml.safe_load((ROOT / "configs/main_benchmark.yaml").read_text()),
            "Prepared scientific configuration changed")
    require(c == read(BASE / "contexts.json")[c["context_id"]],
            "Prepared historical input registration changed")
    setting = m["config"]["models"][model_name]
    require(metadata.version(setting["package"]) == setting["version"], "Wrong model package version")
    for package,version in read(BASE / "numerical_environment.json").items():
        require(metadata.version(package) == version, "Wrong numerical dependency: " + package)
    cp = read(BASE / "checkpoints.json")[model_name]
    require(Path(checkpoint).stat().st_size == cp["bytes"] and sha(checkpoint) == cp["sha256"],
            "Wrong checkpoint; automatic downloads are disabled")
    require(sha(prepared.parent / "query_ids.csv") == m["query_identity_sha256"]
            and sha(prepared.parent / "context_ids.csv") == m["context_identity_sha256"],
            "Prepared row identities changed")
    X, Q, y = [load(m["arrays"][k]) for k in ["X","Q","y"]]
    check_keys = ["y"] if m["preprocessing"].get("reference_available") is False else ["X", "Q", "y"]
    for key in check_keys:
        a = {"X": X, "Q": Q, "y": y}[key]
        require(source_array_sha(a,"depmap") == c[key+"_sha256"], "Prepared numerical input changed")
    import torch
    require(torch.cuda.is_available(), "CUDA unavailable; no scientific CPU fallback")
    require(metadata.version("torch") == "2.11.0+cu128", "Use the historical torch build")
    if model_name == "kumo":
        # Same installed-source check as the unchanged compact Kumo runner.
        import sdm
        package = Path(sdm.__file__).resolve().parent
        source_hashes = ROOT / "data/metadata/sdm_source_hashes.json"
        if source_hashes.is_file():
            for name,digest in read(source_hashes)["sources"].items():
                require(sha(package / name) == digest, "NVIDIA implementation differs")
    out.mkdir(parents=True)
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    kwargs=dict(device="cuda",seed=0,n_estimators=setting["n_estimators"],
                checkpoint=str(checkpoint),categorical_features_indices=[])
    if model_name=="tabpfn":kwargs["tabpfn_options"]=m["config"]["tabpfn"]
    if model_name=="kumo":kwargs["kumo_size"]="large"
    # The C32768 historical runner passed DataFrames with these exact names.
    if model_name!="tabpfn":
        columns=[f"feature_{i}" for i in range(31)]
        X=pd.DataFrame(X,columns=columns);Q=pd.DataFrame(Q,columns=columns)
    try:
        with warnings.catch_warnings(record=True) as caught:
            model=make_model(model_name,**kwargs)
            torch.cuda.synchronize();init=time.perf_counter()-started
            t=time.perf_counter()
            with model_execution(model):model.fit(X,y)
            torch.cuda.synchronize();fit_seconds=time.perf_counter()-t
            t=time.perf_counter()
            prediction=predict_model(model,Q,setting["query_batch_size"])
            torch.cuda.synchronize();prediction_seconds=time.perf_counter()-t
        require(prediction.shape==(c["n_query"],) and np.isfinite(prediction).all(),
                "Invalid prediction vector")
        actual_train=getattr(model,"n_train_samples_",getattr(model,"n_samples_in_",None))
        if model_name=="kumo":actual_train=len(model.x_context_)
        require(actual_train==c["n_train"],"Hidden context-size change")
        actual_estimators=getattr(model,"n_estimators_",getattr(model,"n_estimators",None))
        require(actual_estimators==setting["n_estimators"],"Changed ensemble size")
        record=dict(status="ok",model=model_name,prepared_sha256=sha(prepared),
            historical=c,implementation_sha256=implementation_hashes(),
            checkpoint_sha256=cp["sha256"],package_version=metadata.version(setting["package"]),
            actual_train_samples=int(actual_train),actual_estimators=int(actual_estimators),
            initialization_seconds=init,fit_seconds=fit_seconds,prediction_seconds=prediction_seconds,
            total_model_seconds=time.perf_counter()-started,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
            GPU=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
            warnings=[str(w.message) for w in caught],query_truth_accessed=False,
            original_feature_hash_verified=m["preprocessing"].get("original_feature_hash_verified",True),
            prediction=save(out/"predictions.npy",prediction))
        write(out/"result.json",record)
        return record
    except Exception as exc:
        write(out/"failure.json",dict(status="failed",exception=type(exc).__name__,
            message=str(exc),prepared_sha256=sha(prepared),
            implementation_sha256=implementation_hashes(),query_truth_accessed=False))
        raise


def score(prepared, result, truth, out):
    """Score a finished prediction against the registered held-out query identities."""
    prepared, result, out = Path(prepared), Path(result), Path(out)
    require(not out.exists(), "Scoring output already exists")
    m, r = read(prepared), read(result)
    c = m["historical"]
    require(r["status"] == "ok"
            and r["prepared_sha256"] == sha(prepared)
            and r["implementation_sha256"] == implementation_hashes()
            and r["query_truth_accessed"] is False,
            "Prediction receipt does not match prepared inputs")
    require(sha(prepared.parent / "query_ids.csv") == m["query_identity_sha256"],
            "Query identity file changed")
    ids = pd.read_csv(prepared.parent / "query_ids.csv")
    pairs = pd.MultiIndex.from_frame(ids[["ModelID", "gene"]])
    require(pair_sha(pairs) == c["query_sha256"], "Query order changed")
    raw = next(item for item in read(ROOT / "data/sources.json")["inputs"]
               if item["path"] == "raw/depmap-gene-dependencies.csv")
    require(sha(truth) == raw["sha256"], "Wrong Tahoe target snapshot")
    table = training_targets(truth, sorted(ids.ModelID.unique()),
                             sorted(ids.gene.unique()), parser="pandas")
    target = targets_at(table, pairs)
    reference = read(ROOT / "expected/query_manifest.json")[
        f"depmap/fold{c['fold']}/{c['stratum']}"]
    require(reference["query_sha256"] == c["query_sha256"]
            and array_sha(target) == reference["targets_sha256"],
            "Held-out query targets differ from the registered benchmark")
    predictions = load(r["prediction"])
    require(predictions.shape == target.shape, "Prediction shape differs from targets")
    result = dict(model=m["model"], context_id=c["context_id"],
                  n_train=c["n_train"], n_query=c["n_query"],
                  metrics=metrics(target, predictions), query_sha256=c["query_sha256"],
                  original_feature_hash_verified=r.get("original_feature_hash_verified"),
                  scoring_only_target_access=True)
    write(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", choices=["tabpfn", "tabicl", "kumo"], required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--stratum", choices=STRATA, required=True)
    p.add_argument("--context", type=int, choices=[32768])
    p.add_argument("--pca-mode", choices=["frozen", "refit"], default="frozen")
    p.add_argument("--out", required=True)
    run = sub.add_parser("run")
    run.add_argument("--prepared", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--out", required=True)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--prepared", required=True)
    score_parser.add_argument("--result", required=True)
    score_parser.add_argument("--truth", required=True)
    score_parser.add_argument("--out", required=True)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "prepare":
        result = str(prepare(**args))
    elif action == "run":
        result = run_unit(**args)
    else:
        result = score(**args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
