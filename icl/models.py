"""Adapters for fixed-weight predictors with labelled contexts and held-out queries."""
from contextlib import contextmanager

import numpy as np
import pandas as pd

DEFAULT_KUMO_SIZE = "large"
KUMO_FEATURE_LIMIT = 500


def validate_feature_width(name, width):
    if name == "kumo" and width > KUMO_FEATURE_LIMIT:
        raise ValueError(f"Kumo's default recipe selects at most {KUMO_FEATURE_LIMIT} features per estimator; "
                         f"requested width is {width}; "
                         "choose an explicit full-width recipe or declared feature-subset variant "
                         "before using PCA512 + gene index (513 features)")


def make_model(name, *, device="cpu", seed=0, n_estimators=1,
               checkpoint=None, kumo_size=DEFAULT_KUMO_SIZE, tabpfn_options=None,
               categorical_features_indices=None):
    # Opt-in only: preserve the legacy numerical gene_idx representation.
    categorical_features_indices = list(categorical_features_indices or [])
    if name in ("tabpfn", "tabicl", "kumo") and not checkpoint:
        raise ValueError("Provide an explicit model checkpoint")
    if name == "tabpfn":
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        kwargs = dict(device=device, random_state=seed, n_estimators=n_estimators,
                      n_preprocessing_jobs=1, categorical_features_indices=categorical_features_indices)
        cuda = str(device).startswith("cuda")
        options = {"fit_mode": "fit_with_cache" if cuda else "fit_preprocessors",
                   "keep_cache_on_device": True, "kv_cache_precision": "auto",
                   "inference_precision": "auto", "memory_saving_mode": "auto",
                   "query_execution": "internal" if cuda else "external",
                   "max_batched_test_rows": 8192,
                   "max_batched_estimator_rows": None, "max_batched_estimator_cells": None}
        supplied = tabpfn_options or {}
        if set(supplied) - set(options):
            raise ValueError(f"Unknown TabPFN options: {sorted(set(supplied)-set(options))}")
        options.update(supplied)
        if options["query_execution"] not in ["internal", "external"]:
            raise ValueError("TabPFN query_execution must be internal or external")
        settings = {key: options[key] for key in options if key.startswith("max_batched_")}
        if any(value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0)
               for value in settings.values()):
            raise ValueError("TabPFN batching limits must be nonnegative integers or null")
        if settings["max_batched_test_rows"] is None:
            raise ValueError("max_batched_test_rows must be an integer (0 disables chunking)")
        estimator_options = {k: v for k, v in options.items() if k not in settings and k != "query_execution"}
        precision = estimator_options["inference_precision"]
        if precision in ["float16", "bfloat16", "float32", "float64"]:
            import torch
            estimator_options["inference_precision"] = getattr(torch, precision)
        kwargs.update(estimator_options)
        if checkpoint:
            kwargs["model_path"] = checkpoint
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, **kwargs)
        model.depmap_tabpfn_settings_ = settings
        model.depmap_query_execution_ = options["query_execution"]
        model.depmap_execution_options_ = options
        return model
    if name == "tabicl":
        from tabicl import TabICLRegressor
        return TabICLRegressor(device=device, random_state=seed, n_estimators=n_estimators,
                               n_jobs=1, model_path=checkpoint,
                               checkpoint_version="tabicl-regressor-v2-20260212.ckpt")
    if name == "kumo":
        return KumoRegressor(device=device, seed=seed, n_estimators=n_estimators,
                             size=kumo_size, checkpoint=checkpoint,
                             categorical_features_indices=categorical_features_indices)
    raise ValueError(f"Unknown model: {name}")


class KumoRegressor:
    """Point prediction is the mean of Kumo's 999 predicted quantiles."""
    def __init__(self, device="cpu", seed=0, n_estimators=1, size=DEFAULT_KUMO_SIZE, checkpoint=None,
                 categorical_features_indices=None):
        import sdm
        self.device, self.seed, self.n_estimators = device, seed, n_estimators
        self.categorical_features_indices = list(categorical_features_indices or [])
        if checkpoint:
            import torch
            self.model = sdm.models.KumoTabular(task="regression", size=size,
                                              pretrained=False, device="meta")
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            self.model.models[sdm.Task.regression].load_state_dict(state, assign=True)
            self.model.eval()
        else:
            self.model = sdm.models.KumoTabular(task="regression", size=size, device=device)

    def fit(self, X, y):
        import sdm
        validate_feature_width("kumo", X.shape[1])
        self.columns_ = [f"x{i}" for i in range(X.shape[1])]
        frame = self._frame(X)
        # Its native categorical recipe adds a count column above cardinality 50.
        # Reject a width that would silently drop a feature after that expansion.
        expanded_width = X.shape[1] + sum(frame.iloc[:, i].nunique(dropna=True) > 50
                                        for i in self.categorical_features_indices)
        validate_feature_width("kumo", expanded_width)
        frame["target"] = y
        self.stypes_ = {column: "categorical" if i in self.categorical_features_indices else "numerical"
                        for i, column in enumerate(self.columns_)}
        table = sdm.TableTensor.from_pandas(
            frame, stypes={**self.stypes_, "target": "numerical"}, device=self.device,
        )
        self.x_context_ = table.drop_columns("target")
        self.y_context_ = table[:, "target"]
        return self

    def _frame(self, X):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            frame.columns = self.columns_
            return frame
        return pd.DataFrame(X, columns=self.columns_)

    def predict(self, X):
        import sdm
        import torch
        frame = self._frame(X)
        query = sdm.TableTensor.from_pandas(frame, stypes=self.stypes_, device=self.device)
        with torch.inference_mode():
            out = self.model(x_context=self.x_context_, y_context=self.y_context_, x_query=query,
                             num_estimators=self.n_estimators,
                             generator=torch.Generator(device=self.device).manual_seed(self.seed))
        return out.numerical.mean(dim=-1).cpu().numpy()


def predict_batches(model, X, batch_size):
    if batch_size <= 0 or len(X) == 0:
        raise ValueError("Query matrix and batch size must be nonempty/positive")
    values = [np.asarray(model.predict(X[i:i+batch_size])).reshape(-1)
              for i in range(0, len(X), batch_size)]
    pred = np.concatenate(values)
    if pred.shape != (len(X),) or not np.isfinite(pred).all():
        raise ValueError("Invalid model predictions")
    return pred


@contextmanager
def model_execution(model):
    """Scope TabPFN's process-wide batch settings; project runners are sequential."""
    options = getattr(model, "depmap_tabpfn_settings_", None)
    if options is None:
        yield
        return
    from tabpfn.settings import settings
    previous = {key: getattr(settings.tabpfn, key) for key in options}
    try:
        for key, value in options.items():
            setattr(settings.tabpfn, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(settings.tabpfn, key, value)


def predict_model(model, X, batch_size):
    """TabPFN receives one logical query; other models retain external batches."""
    with model_execution(model):
        if getattr(model, "depmap_query_execution_", "external") == "internal":
            if not len(X):
                raise ValueError("Query matrix must be nonempty")
            pred = np.asarray(model.predict(X)).reshape(-1)
            if pred.shape != (len(X),) or not np.isfinite(pred).all():
                raise ValueError("Invalid model predictions")
            return pred
        return predict_batches(model, X, batch_size)
