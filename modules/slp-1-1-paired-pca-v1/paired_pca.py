"""Streaming paired-assay PCA and a static-feature latent ridge forecast.

The PCA basis is tied to the supplied RNA and protein assay columns. It is an
assay-specific numerical baseline with learned query loadings, not a
vocabulary-free world model or an identified biological state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

Array = np.ndarray
BatchFactory = Callable[[], Iterable[tuple[sparse.csr_matrix, Array]]]
Progress = Callable[[str, int], None]


class PairedPcaError(ValueError):
    """Raised when paired molecular arrays violate the frozen contract."""


@dataclass(frozen=True)
class PairedStats:
    rna_mean: Array
    rna_sd: Array
    protein_mean: Array
    protein_sd: Array
    count: int

    @property
    def rna_weight(self) -> float:
        return 1.0 / np.sqrt(2.0 * self.rna_mean.size)

    @property
    def protein_weight(self) -> float:
        return 1.0 / np.sqrt(2.0 * self.protein_mean.size)

    @property
    def width(self) -> int:
        return self.rna_mean.size + self.protein_mean.size


def _paired(
    rna: sparse.spmatrix | Array,
    protein: Array,
    stats: PairedStats,
) -> tuple[sparse.csr_matrix, Array]:
    matrix = sparse.csr_matrix(rna, dtype=np.float64)
    values = np.asarray(protein, dtype=np.float64)
    if matrix.ndim != 2 or values.ndim != 2 or matrix.shape[0] != values.shape[0]:
        raise PairedPcaError("RNA and protein rows must align")
    if matrix.shape[1] != stats.rna_mean.size or values.shape[1] != stats.protein_mean.size:
        raise PairedPcaError("paired query widths differ from fitted statistics")
    if not np.isfinite(matrix.data).all() or not np.isfinite(values).all():
        raise PairedPcaError("paired molecular values must be finite")
    if np.any(stats.rna_sd <= 0) or np.any(stats.protein_sd <= 0):
        raise PairedPcaError("fitted standard deviations must be positive")
    return matrix, values


def fit_stats(factory: BatchFactory, *, floor: float = 0.05) -> PairedStats:
    """Fit the exact train-cell population mean and SD in two streaming sums."""
    if not np.isfinite(floor) or floor <= 0:
        raise PairedPcaError("SD floor must be finite and positive")
    rna_sum = rna_square = protein_sum = protein_square = None
    count = 0
    for raw_rna, raw_protein in factory():
        rna = sparse.csr_matrix(raw_rna, dtype=np.float64)
        protein = np.asarray(raw_protein, dtype=np.float64)
        if rna.shape[0] != protein.shape[0] or not np.isfinite(rna.data).all() or not np.isfinite(protein).all():
            raise PairedPcaError("invalid paired statistics batch")
        if rna_sum is None:
            rna_sum = np.zeros(rna.shape[1], dtype=np.float64)
            rna_square = np.zeros(rna.shape[1], dtype=np.float64)
            protein_sum = np.zeros(protein.shape[1], dtype=np.float64)
            protein_square = np.zeros(protein.shape[1], dtype=np.float64)
        if rna.shape[1] != rna_sum.size or protein.shape[1] != protein_sum.size:
            raise PairedPcaError("query width changes between batches")
        rna_sum += np.asarray(rna.sum(axis=0)).ravel()
        rna_square += np.asarray(rna.multiply(rna).sum(axis=0)).ravel()
        protein_sum += protein.sum(axis=0)
        protein_square += np.square(protein).sum(axis=0)
        count += rna.shape[0]
    if not count or rna_sum is None:
        raise PairedPcaError("statistics factory produced no cells")
    rna_mean = rna_sum / count
    protein_mean = protein_sum / count
    rna_sd = np.maximum(np.sqrt(np.maximum(rna_square / count - rna_mean**2, 0.0)), floor)
    protein_sd = np.maximum(
        np.sqrt(np.maximum(protein_square / count - protein_mean**2, 0.0)), floor,
    )
    # This float32 freeze matches the paired AE's train-cell normalization.
    return PairedStats(
        rna_mean.astype(np.float32), rna_sd.astype(np.float32),
        protein_mean.astype(np.float32), protein_sd.astype(np.float32), count,
    )


def affine_right(
    raw_rna: sparse.spmatrix | Array,
    raw_protein: Array,
    right: Array,
    stats: PairedStats,
) -> Array:
    """Compute balanced centered-standardized paired input times ``right``."""
    rna, protein = _paired(raw_rna, raw_protein, stats)
    matrix = np.asarray(right, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != stats.width or not np.isfinite(matrix).all():
        raise PairedPcaError("right matrix does not align with paired input")
    split = stats.rna_mean.size
    rna_right, protein_right = matrix[:split], matrix[split:]
    rna_multiplier = stats.rna_weight / stats.rna_sd
    protein_multiplier = stats.protein_weight / stats.protein_sd
    result = rna @ (rna_right * rna_multiplier[:, None])
    result -= (stats.rna_mean * rna_multiplier) @ rna_right
    result += (protein - stats.protein_mean) @ (
        protein_right * protein_multiplier[:, None]
    )
    return np.asarray(result, dtype=np.float64)


def affine_transpose(
    raw_rna: sparse.spmatrix | Array,
    raw_protein: Array,
    right: Array,
    stats: PairedStats,
) -> Array:
    """Compute balanced centered-standardized paired input transpose times right."""
    rna, protein = _paired(raw_rna, raw_protein, stats)
    matrix = np.asarray(right, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != rna.shape[0] or not np.isfinite(matrix).all():
        raise PairedPcaError("transpose right matrix must align with cell rows")
    total = matrix.sum(axis=0)
    rna_result = (
        np.asarray(rna.T @ matrix) - stats.rna_mean[:, None] * total[None, :]
    ) * (stats.rna_weight / stats.rna_sd)[:, None]
    protein_result = (
        (protein - stats.protein_mean).T @ matrix
    ) * (stats.protein_weight / stats.protein_sd)[:, None]
    return np.vstack((rna_result, protein_result))


@dataclass(frozen=True)
class StreamingPca:
    components: Array
    eigenvalues: Array
    stats: PairedStats
    passes: int
    oversample: int
    seed: int

    def encode(self, raw_rna: sparse.spmatrix | Array, raw_protein: Array) -> Array:
        return affine_right(raw_rna, raw_protein, self.components, self.stats)

    def reconstruct_balanced(self, scores: Array) -> Array:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.components.shape[1]:
            raise PairedPcaError("scores do not align with PCA components")
        return values @ self.components.T

    def decode_raw(self, scores: Array) -> tuple[Array, Array]:
        balanced = self.reconstruct_balanced(scores)
        split = self.stats.rna_mean.size
        rna = (
            balanced[:, :split] / self.stats.rna_weight * self.stats.rna_sd
            + self.stats.rna_mean
        )
        protein = (
            balanced[:, split:] / self.stats.protein_weight * self.stats.protein_sd
            + self.stats.protein_mean
        )
        return rna, protein

    def decode_delta(self, scores: Array) -> tuple[Array, Array]:
        balanced = self.reconstruct_balanced(scores)
        split = self.stats.rna_mean.size
        return (
            balanced[:, :split] / self.stats.rna_weight * self.stats.rna_sd,
            balanced[:, split:] / self.stats.protein_weight * self.stats.protein_sd,
        )


def fit_streaming_pca(
    factory: BatchFactory,
    stats: PairedStats,
    *,
    rank: int = 128,
    oversample: int = 32,
    passes: int = 3,
    seed: int = 731,
    progress: Progress | None = None,
) -> StreamingPca:
    """Fit the frozen covariance subspace iteration without dense cell storage."""
    if min(rank, oversample, passes) <= 0 or rank + oversample > stats.width:
        raise PairedPcaError("invalid PCA rank, oversampling, or pass count")
    generator = np.random.default_rng(seed)
    q, _ = np.linalg.qr(generator.normal(size=(stats.width, rank + oversample)))
    for iteration in range(passes):
        product = np.zeros_like(q)
        cells = 0
        for rna, protein in factory():
            projected = affine_right(rna, protein, q, stats)
            product += affine_transpose(rna, protein, projected, stats)
            cells += len(protein)
        if cells != stats.count:
            raise PairedPcaError("PCA pass cell count differs from fitted statistics")
        q, _ = np.linalg.qr(product)
        if progress is not None:
            progress("subspace", iteration + 1)
    rayleigh = np.zeros((q.shape[1], q.shape[1]), dtype=np.float64)
    cells = 0
    for rna, protein in factory():
        projected = affine_right(rna, protein, q, stats)
        rayleigh += projected.T @ projected
        cells += len(protein)
    if cells != stats.count:
        raise PairedPcaError("Rayleigh pass cell count differs from fitted statistics")
    if progress is not None:
        progress("rayleigh", passes + 1)
    eigenvalues, eigenvectors = np.linalg.eigh((rayleigh + rayleigh.T) * 0.5)
    order = np.argsort(eigenvalues)[::-1][:rank]
    components = q @ eigenvectors[:, order]
    return StreamingPca(components, eigenvalues[order], stats, passes, oversample, seed)


@dataclass(frozen=True)
class LatentRidge:
    feature_mean: Array
    feature_scale: Array
    feature_clip: float
    coefficient: Array
    intercept: Array
    alpha: float

    def normalize(self, features: Array) -> Array:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_mean.size:
            raise PairedPcaError("action feature shape mismatch")
        if not np.isfinite(values).all():
            raise PairedPcaError("action features must be finite")
        return np.clip(
            (values - self.feature_mean) / self.feature_scale,
            -self.feature_clip,
            self.feature_clip,
        )

    def predict(self, features: Array, context_index: Array) -> Array:
        x = self.normalize(features)
        context = np.asarray(context_index, dtype=np.int64)
        if context.shape != (len(x),) or np.any(context < 0) or np.any(context >= len(self.coefficient)):
            raise PairedPcaError("context indices are invalid")
        result = np.empty((len(x), self.intercept.shape[1]), dtype=np.float64)
        for index in range(len(self.coefficient)):
            rows = context == index
            result[rows] = x[rows] @ self.coefficient[index] + self.intercept[index]
        return result


def fit_latent_ridge(
    features: Array,
    targets: Array,
    *,
    alpha: float = 10_000.0,
) -> tuple[Array, Array]:
    """Fit one intercept-bearing ridge through the sample-space dual."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or not len(x):
        raise PairedPcaError("latent ridge features and targets must align")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or alpha <= 0:
        raise PairedPcaError("latent ridge inputs and alpha must be finite")
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    centered_x, centered_y = x - x_mean, y - y_mean
    dual = np.linalg.solve(centered_x @ centered_x.T + alpha * np.eye(len(x)), centered_y)
    coefficient = centered_x.T @ dual
    intercept = y_mean - x_mean @ coefficient
    return coefficient, intercept


@dataclass(frozen=True)
class PcaForecastArtifact:
    pca: StreamingPca
    ridge: LatentRidge
    rna_controls: Array
    protein_controls: Array
    context_names: Array

    def forecast(
        self,
        action_features: Array,
        context_index: Array,
    ) -> tuple[Array, Array, Array]:
        context = np.asarray(context_index, dtype=np.int64)
        delta = self.ridge.predict(action_features, context)
        rna_delta, protein_delta = self.pca.decode_delta(delta)
        return (
            self.rna_controls[context] + rna_delta,
            self.protein_controls[context] + protein_delta,
            delta,
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            model_type=np.asarray("paired-assay-pca128-static-latent-ridge-v1"),
            components=self.pca.components,
            eigenvalues=self.pca.eigenvalues,
            rna_mean=self.pca.stats.rna_mean,
            rna_sd=self.pca.stats.rna_sd,
            protein_mean=self.pca.stats.protein_mean,
            protein_sd=self.pca.stats.protein_sd,
            fitting_cell_count=np.asarray(self.pca.stats.count),
            passes=np.asarray(self.pca.passes),
            oversample=np.asarray(self.pca.oversample),
            seed=np.asarray(self.pca.seed),
            feature_mean=self.ridge.feature_mean,
            feature_scale=self.ridge.feature_scale,
            feature_clip=np.asarray(self.ridge.feature_clip),
            ridge_coefficient=self.ridge.coefficient,
            ridge_intercept=self.ridge.intercept,
            ridge_alpha=np.asarray(self.ridge.alpha),
            rna_controls=self.rna_controls,
            protein_controls=self.protein_controls,
            context_names=self.context_names,
        )

    @classmethod
    def load(cls, path: Path) -> PcaForecastArtifact:
        with np.load(path, allow_pickle=False) as item:
            if str(item["model_type"]) != "paired-assay-pca128-static-latent-ridge-v1":
                raise PairedPcaError("artifact model type mismatch")
            stats = PairedStats(
                item["rna_mean"], item["rna_sd"], item["protein_mean"],
                item["protein_sd"], int(item["fitting_cell_count"]),
            )
            pca = StreamingPca(
                item["components"], item["eigenvalues"], stats,
                int(item["passes"]), int(item["oversample"]), int(item["seed"]),
            )
            ridge = LatentRidge(
                item["feature_mean"], item["feature_scale"], float(item["feature_clip"]),
                item["ridge_coefficient"], item["ridge_intercept"], float(item["ridge_alpha"]),
            )
            return cls(
                pca, ridge, item["rna_controls"], item["protein_controls"],
                item["context_names"],
            )


def assert_held_genes_excluded(training_action_ids: Array, held_action_ids: Array) -> None:
    training = {str(item) for item in np.asarray(training_action_ids) if str(item)}
    held = {str(item) for item in np.asarray(held_action_ids) if str(item)}
    overlap = sorted(training & held)
    if overlap:
        raise PairedPcaError(f"held genes occur in representation fitting rows: {overlap[:3]}")
