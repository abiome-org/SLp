"""Augment a frozen gene-state graph with frozen response-query descriptors."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

Array = np.ndarray
STATIC_WIDTH = 577
RESPONSE_WIDTH = 32
NODE_WIDTH = 610
TAXON = 9606


class GeneStateFeatureError(ValueError):
    """Raised when graph or reference identities do not align exactly."""


def _fields(source: object) -> set[str]:
    if hasattr(source, "files"):
        return set(source.files)  # type: ignore[attr-defined]
    return set(source.keys())  # type: ignore[attr-defined]


def augment_graph(
    graph: object, reference: object
) -> tuple[dict[str, Array], dict[str, object]]:
    """Copy every graph field and add normalized response descriptors by ENSG ID."""

    required_graph = {
        "node_ids",
        "node_taxon",
        "static_features",
        "static_feature_observed",
        "feature_mean",
        "feature_scale",
        "adjacency_indptr",
        "adjacency_indices",
        "adjacency_weights",
        "action_node_index",
        "query_node_index",
    }
    required_reference = {
        "query_ids",
        "query_features",
        "query_feature_mean",
        "query_feature_std",
    }
    if required_graph - _fields(graph) or required_reference - _fields(reference):
        raise GeneStateFeatureError("graph or reference fields are incomplete")
    node_ids = graph["node_ids"].astype(str)
    query_ids = reference["query_ids"].astype(str)
    if len(set(node_ids)) != len(node_ids) or len(set(query_ids)) != len(query_ids):
        raise GeneStateFeatureError("node and reference query IDs must be unique")
    if graph["static_features"].shape != (len(node_ids), STATIC_WIDTH):
        raise GeneStateFeatureError("expected normalized static feature width 577")
    if graph["node_taxon"].shape != (len(node_ids),) or np.any(
        graph["node_taxon"] != TAXON
    ):
        raise GeneStateFeatureError("graph taxonomy contract drift")

    query_features = np.asarray(reference["query_features"], dtype=np.float64)
    query_mean = np.asarray(reference["query_feature_mean"], dtype=np.float64)
    query_std = np.asarray(reference["query_feature_std"], dtype=np.float64)
    if (
        query_features.ndim != 2
        or query_features.shape[0] != len(query_ids)
        or query_features.shape[1] < RESPONSE_WIDTH
        or query_mean.shape != (query_features.shape[1],)
        or query_std.shape != query_mean.shape
        or not np.isfinite(query_features).all()
        or not np.isfinite(query_mean).all()
        or not np.isfinite(query_std).all()
        or np.any(query_std[-RESPONSE_WIDTH:] <= 0)
    ):
        raise GeneStateFeatureError("reference query feature contract drift")

    node_lookup = {value: index for index, value in enumerate(node_ids)}
    unmatched = [value for value in query_ids if value not in node_lookup]
    if unmatched:
        raise GeneStateFeatureError(
            f"reference queries absent from graph: {unmatched[:3]}"
        )
    graph_query_ids = node_ids[np.asarray(graph["query_node_index"], dtype=np.int64)]
    if not np.array_equal(graph_query_ids, query_ids):
        raise GeneStateFeatureError(
            "graph query mapping and reference query order differ"
        )

    response = (
        query_features[:, -RESPONSE_WIDTH:] - query_mean[-RESPONSE_WIDTH:]
    ) / query_std[-RESPONSE_WIDTH:]
    if not np.isfinite(response).all():
        raise GeneStateFeatureError("normalized response descriptors are nonfinite")
    node_features = np.zeros((len(node_ids), NODE_WIDTH), dtype=np.float32)
    node_features[:, :STATIC_WIDTH] = graph["static_features"]
    available = np.zeros(len(node_ids), dtype=bool)
    for query_index, query_id in enumerate(query_ids):
        node_index = node_lookup[query_id]
        node_features[node_index, STATIC_WIDTH : STATIC_WIDTH + RESPONSE_WIDTH] = (
            response[query_index].astype(np.float32)
        )
        node_features[node_index, -1] = 1.0
        available[node_index] = True
    if np.count_nonzero(node_features[~available, STATIC_WIDTH:]) != 0:
        raise GeneStateFeatureError(
            "missing-query nodes must have zero response block and flag"
        )
    if not np.array_equal(node_features[:, :STATIC_WIDTH], graph["static_features"]):
        raise GeneStateFeatureError("static feature block changed")

    arrays = {name: graph[name] for name in graph.files}  # type: ignore[attr-defined]
    arrays.update(
        {
            "node_features": node_features,
            "response_query_feature_observed": available,
            "node_feature_block_widths": np.asarray(
                [STATIC_WIDTH, RESPONSE_WIDTH, 1], dtype=np.int64
            ),
            "response_query_feature_mean": reference["query_feature_mean"][
                -RESPONSE_WIDTH:
            ],
            "response_query_feature_std": reference["query_feature_std"][
                -RESPONSE_WIDTH:
            ],
        }
    )
    audit = {
        "nodes": len(node_ids),
        "node_feature_shape": list(node_features.shape),
        "response_descriptor_nodes": int(available.sum()),
        "missing_response_descriptor_nodes": int((~available).sum()),
        "all_reference_queries_matched_once": True,
        "graph_query_mapping_exact": True,
        "static_block_exact": True,
        "response_normalization": "(reference.query_features[-32] - reference.query_feature_mean[-32]) / reference.query_feature_std[-32]",
        "response_provenance": "quantitative fitting-derived RNA response-query descriptors; not static priors",
    }
    return arrays, audit


def array_sha256(value: Array) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def graph_field_audit(
    source: object, augmented: object
) -> dict[str, dict[str, object]]:
    """Return logical-byte hashes proving every original graph field is unchanged."""

    report = {}
    for name in source.files:  # type: ignore[attr-defined]
        source_value = source[name]
        augmented_value = augmented[name]
        source_hash = array_sha256(source_value)
        augmented_hash = array_sha256(augmented_value)
        report[name] = {
            "exact": bool(
                source_value.dtype == augmented_value.dtype
                and source_value.shape == augmented_value.shape
                and source_hash == augmented_hash
            ),
            "dtype": str(source_value.dtype),
            "shape": list(source_value.shape),
            "source_sha256": source_hash,
            "augmented_sha256": augmented_hash,
        }
    return report


def write_npz(path: Path, arrays: dict[str, Array]) -> None:
    """Write deterministic, scalar-preserving, pickle-free NPZ."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for name in sorted(arrays):
                stream = io.BytesIO()
                value = arrays[name]
                payload = value if value.ndim == 0 else np.ascontiguousarray(value)
                np.save(stream, payload, allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, stream.getvalue())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
