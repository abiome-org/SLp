"""Produce byte-level verification for the immutable four-context snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and C-order logical bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left_hash = array_sha256(left)
    right_hash = array_sha256(right)
    return {
        "exact": bool(
            left.dtype == right.dtype
            and left.shape == right.shape
            and left_hash == right_hash
        ),
        "dtype": str(left.dtype),
        "shape": list(left.shape),
        "source_sha256": left_hash,
        "merged_sha256": right_hash,
    }


def logical_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    """Compare values when a combined Unicode field necessarily has a wider dtype."""

    exact = bool(left.shape == right.shape and np.array_equal(left, right))

    def value_hash(value: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        for item in value.astype(str).flat:
            encoded = item.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        return digest.hexdigest()

    return {
        "exact": exact,
        "source_dtype": str(left.dtype),
        "merged_dtype": str(right.dtype),
        "shape": list(left.shape),
        "source_value_sha256": value_hash(left),
        "merged_value_sha256": value_hash(right),
    }


def verify(replogle_path: Path, hepg2_path: Path, merged_path: Path) -> dict:
    with (
        np.load(replogle_path, allow_pickle=False) as replogle,
        np.load(hepg2_path, allow_pickle=False) as hepg2,
        np.load(merged_path, allow_pickle=False) as merged,
    ):
        original_rows = len(replogle["action_ids"])
        roles = hepg2["split_role"].astype(str)
        selected = np.flatnonzero(roles != "test")
        prefix = {}
        for field in (
            "targets",
            "observed",
            "action_ids",
            "context_index",
            "record_ids",
            "num_cells_filtered",
        ):
            prefix[field] = comparison(replogle[field], merged[field][:original_rows])
        for field in ("query_ids", "context_value_space", "num_cells_role"):
            prefix[field] = comparison(replogle[field], merged[field])
        for field in (
            "context_ids",
            "basal_control",
            "context_basal_expression",
            "context_basal_observed",
        ):
            prefix[field] = comparison(replogle[field], merged[field][:3])
        for field in (
            "control_targets",
            "control_observed",
            "control_context_index",
            "control_num_cells_filtered",
            "control_record_ids",
            "control_core",
        ):
            prefix[field] = comparison(replogle[field], merged[field])
        prefix["split_train"] = comparison(
            replogle["split_train"],
            merged["split_train"][: len(replogle["split_train"])],
        )
        prefix["split_validation"] = comparison(
            replogle["split_validation"],
            merged["split_validation"][: len(replogle["split_validation"])],
        )

        suffix = {}
        for target, source in (
            ("targets", "targets"),
            ("observed", "observed"),
            ("action_ids", "action_ids"),
            ("record_ids", "record_ids"),
            ("num_cells_filtered", "num_cells_filtered"),
            ("source_population_ids", "source_population_ids"),
            ("source_construct_ids", "source_construct_ids"),
            ("source_transcript_labels", "source_transcript_labels"),
        ):
            source_value = hepg2[source][selected]
            if target == "num_cells_filtered":
                source_value = source_value.astype(np.float32)
            merged_value = merged[target][original_rows:]
            suffix[target] = (
                logical_comparison(source_value.astype(str), merged_value.astype(str))
                if source_value.dtype.kind in "US"
                else comparison(source_value, merged_value)
            )
        suffix["query_num_cells_filtered"] = comparison(
            hepg2["query_num_cells_filtered"][selected],
            merged["hepg2_query_num_cells_filtered"],
        )

        arrays_pickle_free = all(
            merged[name].dtype.kind != "O" for name in merged.files
        )
        train = merged["split_train"]
        validation = merged["split_validation"]
        train_actions = set(merged["action_ids"][train].astype(str).tolist())
        validation_actions = set(merged["action_ids"][validation].astype(str).tolist())
        no_test_role = bool(np.all(merged["split_role"].astype(str) != "test"))
        checks = {
            "all_replogle_prefixes_exact": all(
                value["exact"] for value in prefix.values()
            ),
            "all_selected_hepg2_suffixes_exact": all(
                value["exact"] for value in suffix.values()
            ),
            "train_validation_action_overlap": len(train_actions & validation_actions),
            "test_roles_in_output": int(
                np.sum(merged["split_role"].astype(str) == "test")
            ),
            "excluded_hepg2_test_records": int(np.sum(roles == "test")),
            "all_arrays_pickle_free": arrays_pickle_free,
            "hepg2_observed_matches_query_counts": bool(
                np.array_equal(
                    merged["observed"][original_rows:],
                    merged["hepg2_query_num_cells_filtered"] > 0,
                )
            ),
            "hepg2_control_target_pseudobulks_available": bool(
                merged["control_target_pseudobulks_available_by_context"][3]
            ),
            "hepg2_control_stats_compatible_with_replogle_control_targets": bool(
                merged["hepg2_control_stats_compatible_with_control_targets"].item()
            ),
        }
        if not (
            checks["all_replogle_prefixes_exact"]
            and checks["all_selected_hepg2_suffixes_exact"]
            and checks["train_validation_action_overlap"] == 0
            and no_test_role
            and checks["all_arrays_pickle_free"]
            and checks["hepg2_observed_matches_query_counts"]
            and not checks["hepg2_control_target_pseudobulks_available"]
            and not checks[
                "hepg2_control_stats_compatible_with_replogle_control_targets"
            ]
        ):
            raise ValueError(f"four-context verification failed: {checks}")
        return {
            "schema": "slp.human-four-context-development-verification/v1",
            "development": {
                "sha256": file_sha256(merged_path),
                "bytes": merged_path.stat().st_size,
            },
            "replogle_prefix": prefix,
            "hepg2_selected_suffix": suffix,
            "checks": checks,
            "counts": {
                "records": len(merged["action_ids"]),
                "train": len(train),
                "validation": len(validation),
                "queries": len(merged["query_ids"]),
                "contexts": len(merged["context_ids"]),
            },
            "target_spaces": merged["target_value_space_by_context"]
            .astype(str)
            .tolist(),
            "hepg2_control_semantics": {
                "target_baseline": "zero by per-GEM control centering for supported queries",
                "target_baseline_observed_queries": int(
                    merged["basal_control_observed"][3].sum()
                ),
                "independent_control_target_pseudobulks": "unavailable",
                "retained_stats": "control-only per-GEM linear normalization mean/std/support and cell counts",
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replogle", type=Path, required=True)
    parser.add_argument("--hepg2", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.replogle, args.hepg2, args.merged)
    report["verification_source_sha256"] = file_sha256(Path(__file__))
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
