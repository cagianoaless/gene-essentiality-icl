"""Extracted verified training-only expression preprocessing."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def unique(values, name):
    index = pd.Index(values)
    if index.hasnans or not index.is_unique:
        raise ValueError(f"Missing or duplicated {name}")
    return index

@dataclass
class ExpressionPCA:
    dimensions: int
    seed: int = 0

    def fit(self, counts, fit_ids):
        unique(fit_ids, "PCA fitting cells")
        if len(counts) != len(fit_ids):
            raise ValueError("PCA fitting IDs do not match rows")
        if self.dimensions not in [15, 64, 128, 256, 512] or self.dimensions > min(counts.shape[0] - 1, counts.shape[1]):
            raise ValueError("Requested PCA dimension is unsupported or exceeds matrix rank bound")
        totals = counts.sum(axis=1, dtype=np.float64)
        if (totals <= 0).any():
            raise ValueError("Zero-count cell line")
        self.target_sum_ = float(np.median(totals))
        self.fit_ids_ = tuple(fit_ids)
        self.model_ = PCA(n_components=self.dimensions, svd_solver="randomized", random_state=self.seed)
        self.model_.fit(self.normalize(counts))
        return self

    def normalize(self, counts):
        totals = counts.sum(axis=1, dtype=np.float64)
        if (totals <= 0).any():
            raise ValueError("Zero-count cell line")
        return np.log1p(counts * (self.target_sum_ / totals[:, None])).astype(np.float32)

    def transform(self, counts):
        result = self.model_.transform(self.normalize(counts)).astype(np.float32)
        if result.shape != (len(counts), self.dimensions) or not np.isfinite(result).all():
            raise ValueError("Invalid PCA output")
        return result

    def gene_loadings(self):
        """Expression-derived coordinates from this exact fitted cell PCA."""
        result = self.model_.components_.T * np.sqrt(self.model_.explained_variance_)
        if result.shape != (self.model_.n_features_in_, self.dimensions) or not np.isfinite(result).all():
            raise ValueError("Invalid gene loadings")
        return result

    def assert_training_only(self, train, heldout):
        if set(self.fit_ids_) != set(train) or set(self.fit_ids_) & set(heldout):
            raise ValueError("PCA was not fitted exclusively on intended training cells")
