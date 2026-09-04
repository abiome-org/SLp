"""Synthetic determinism and fail-closed tests for the SGD mapping module."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-sgd-map"
sys.path.insert(0, str(MODULE))

from mapper import (  # noqa: E402
    FileSpec,
    MapBounds,
    README_MARKERS,
    SgdMapError,
    canonical_mapping_digest,
    normalize_sgd_snapshot,
    resolve_pinned_dataset_input,
)


class SgdMapTest(unittest.TestCase):
    @staticmethod
    def _feature(
        primary: str,
        systematic: str,
        standard: str = "",
        aliases: str = "",
        *,
        feature_type: str = "ORF",
    ) -> str:
        return "\t".join(
            [
                primary,
                feature_type,
                "Verified" if feature_type == "ORF" else "",
                systematic,
                standard,
                aliases,
                "chromosome 1",
                "",
                "1",
                "1",
                "10",
                "W",
                "",
                "2024-05-28",
                "2024-05-28",
                "synthetic description",
            ]
        )

    @staticmethod
    def _xref(
        accession: str,
        source: str,
        accession_type: str,
        feature_name: str,
        primary: str,
        display: str = "",
    ) -> str:
        return "\t".join(
            [accession, source, accession_type, feature_name, primary, display]
        )

    @staticmethod
    def _retired(primary: str = "S000000010") -> str:
        return "\t".join(
            [
                "YAL010C",
                "ORF|Merged",
                "1",
                "100",
                "200",
                "C",
                primary,
                "",
                "YAL011W",
                "S000000011",
                "synthetic retired description",
                "synthetic annotation note",
                "2006-01-01",
            ]
        )

    def _write_snapshot(
        self,
        root: Path,
        *,
        feature_lines: list[str] | None = None,
        xref_lines: list[str] | None = None,
        retired_lines: list[str] | None = None,
        readme_overrides: dict[str, str] | None = None,
    ) -> tuple[Path, tuple[FileSpec, ...]]:
        snapshot = root / "raw"
        snapshot.mkdir(parents=True)
        features = feature_lines or [
            self._feature("S000000002", "YAL043C-a"),
            "",
            self._feature("S000000001", "YAL001C-A", "TFC3", "FUN24|YAL001C"),
            self._feature("S000000003", "YAL043C-A"),
            self._feature(
                "S000000004", "tA(AGC)A", feature_type="tRNA gene"
            ),
        ]
        xrefs = xref_lines or [
            self._xref("P12345", "SIB", "Swiss-Prot ID", "YAL001C-A", "S000000001", "TFC3"),
            self._xref("P12345", "SIB", "Swiss-Prot ID", "YAL043C-a", "S000000002"),
            self._xref("P12345", "Other", "Opaque ID", "YAL043C-A", "S000000003"),
            self._xref("R00001", "Archive", "Retired ID", "YAL010C", "S000000010"),
            self._xref("N00001", "NCBI", "RNA ID", "tA(AGC)A", "S000000004"),
            self._xref("U00001", "Other", "Unscoped SGD ID", "YSC0001", "S000000099"),
        ]
        retired = retired_lines or [self._retired(), "broken\tphysical-row"]
        contents = {
            "SGD_features.tab": "\n".join(features) + "\n",
            "dbxref.tab": "\n".join(xrefs) + "\n",
            "deleted_merged_features.tab": "\n".join(retired) + "\n",
        }
        for name, markers in README_MARKERS.items():
            contents[name] = "\n".join(markers) + "\n"
        if readme_overrides:
            contents.update(readme_overrides)
        for name, content in contents.items():
            (snapshot / name).write_text(content, encoding="utf-8", newline="\n")

        specs: list[FileSpec] = []
        for name in sorted(contents):
            payload = (snapshot / name).read_bytes()
            options: dict[str, int] = {}
            if name.endswith(".tab"):
                lines = payload.splitlines()
                options["physical_lines"] = len(lines)
                if name == "SGD_features.tab":
                    options["data_records"] = sum(bool(line) for line in lines)
                    options["irregular_records"] = sum(not line for line in lines)
                elif name == "dbxref.tab":
                    options["data_records"] = len(lines)
                    options["irregular_records"] = 0
                else:
                    options["data_records"] = sum(
                        len(line.split(b"\t")) == 13 for line in lines
                    )
                    options["irregular_records"] = sum(
                        len(line.split(b"\t")) != 13 for line in lines
                    )
            specs.append(
                FileSpec(
                    name=name,
                    bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    **options,
                )
            )
        return snapshot, tuple(specs)

    @staticmethod
    def _jsonl(path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_deterministic_exact_identity_and_one_to_many_relations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_raw, specs = self._write_snapshot(root / "one")
            second_raw, second_specs = self._write_snapshot(root / "two")
            first = root / "first-output"
            second = root / "second-output"
            report_one = normalize_sgd_snapshot(
                first_raw, first, MapBounds(), file_specs=specs
            )
            report_two = normalize_sgd_snapshot(
                second_raw, second, MapBounds(), file_specs=second_specs
            )
            self.assertEqual(
                report_one["identityMappingSha256"],
                report_two["identityMappingSha256"],
            )
            for name in (
                "current-orfs.jsonl",
                "external-accessions.jsonl",
                "retired-merged-quarantine.jsonl",
                "mapping-manifest.json",
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            orfs = self._jsonl(first / "current-orfs.jsonl")
            self.assertEqual(
                [item["systematicName"] for item in orfs],
                ["YAL001C-A", "YAL043C-a", "YAL043C-A"],
            )
            self.assertEqual(orfs[0]["displayMetadata"]["standardGeneName"], "TFC3")
            self.assertIs(orfs[0]["displayMetadata"]["resolvesIdentity"], False)
            self.assertIs(orfs[0]["secondaryIdentifiersResolve"], False)

            relations = self._jsonl(first / "external-accessions.jsonl")
            swiss = next(
                item
                for item in relations
                if item["typedAccession"]
                == {
                    "value": "P12345",
                    "source": "SIB",
                    "type": "Swiss-Prot ID",
                    "caseNormalization": "none",
                    "namespaceInferred": False,
                }
            )
            self.assertEqual(swiss["targetCount"], 2)
            self.assertEqual(
                [target["canonicalSgdCurie"] for target in swiss["targets"]],
                ["SGD:S000000001", "SGD:S000000002"],
            )
            self.assertIs(swiss["relationOnly"], True)
            statuses = {
                item["typedAccession"]["value"]: item["targets"][0]["targetStatus"]
                for item in relations
                if item["typedAccession"]["value"] in {"R00001", "N00001", "U00001"}
            }
            self.assertEqual(
                statuses,
                {
                    "R00001": "retired-or-merged",
                    "N00001": "current-non-orf",
                    "U00001": "not-in-current-feature-table",
                },
            )

            quarantine = self._jsonl(first / "retired-merged-quarantine.jsonl")
            self.assertEqual([item["recordKind"] for item in quarantine], [
                "retired-or-merged",
                "malformed-source-row",
            ])
            self.assertIs(quarantine[0]["automaticRedirectAllowed"], False)
            self.assertIs(
                quarantine[0]["reportedReplacement"]["evidenceOnly"], True
            )
            manifest = json.loads((first / "mapping-manifest.json").read_text())
            self.assertEqual(
                canonical_mapping_digest(manifest["digestBasis"]),
                manifest["identityMappingSha256"],
            )
            self.assertEqual(report_one["oneToManyExternalRelationCount"], 1)

    def test_legacy_retired_primary_is_preserved_without_fabricated_curie(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, specs = self._write_snapshot(
                root, retired_lines=[self._retired("L000003336")]
            )
            output = root / "output"
            normalize_sgd_snapshot(raw, output, MapBounds(), file_specs=specs)
            record = self._jsonl(output / "retired-merged-quarantine.jsonl")[0]
            self.assertEqual(record["sourcePrimaryIdentifier"], "L000003336")
            self.assertIsNone(record["retiredPrimarySgdCurie"])
            self.assertIn(
                "noncanonical-legacy-primary-identifier",
                record["sourceContractIssues"],
            )

    def test_exact_file_set_bytes_and_sha256_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, specs = self._write_snapshot(root)
            feature = raw / "SGD_features.tab"
            payload = bytearray(feature.read_bytes())
            payload[0] = ord("T")
            feature.write_bytes(payload)
            with self.assertRaisesRegex(SgdMapError, "SHA-256 drift"):
                normalize_sgd_snapshot(raw, root / "output", MapBounds(), file_specs=specs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, specs = self._write_snapshot(root)
            (raw / "extra.tab").write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(SgdMapError, "file set drift"):
                normalize_sgd_snapshot(raw, root / "output", MapBounds(), file_specs=specs)

    def test_readme_and_six_column_payload_contracts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, specs = self._write_snapshot(
                root,
                readme_overrides={"dbxref.README": "not the pinned column contract\n"},
            )
            with self.assertRaisesRegex(SgdMapError, "README column contract drift"):
                normalize_sgd_snapshot(raw, root / "output", MapBounds(), file_specs=specs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, specs = self._write_snapshot(
                root,
                xref_lines=["\t".join(["P12345", "SIB", "Swiss-Prot ID", "YAL001C-A", "S000000001"])],
            )
            with self.assertRaisesRegex(SgdMapError, "exactly six columns"):
                normalize_sgd_snapshot(raw, root / "output", MapBounds(), file_specs=specs)

    def test_case_sensitive_systematic_keys_are_not_collapsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, specs = self._write_snapshot(root)
            output = root / "output"
            normalize_sgd_snapshot(raw, output, MapBounds(), file_specs=specs)
            names = {item["systematicName"] for item in self._jsonl(output / "current-orfs.jsonl")}
            self.assertIn("YAL043C-a", names)
            self.assertIn("YAL043C-A", names)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = [
                self._feature("S000000001", "YAL001C"),
                self._feature("S000000002", "YAL001C"),
            ]
            raw, specs = self._write_snapshot(root, feature_lines=duplicate)
            with self.assertRaisesRegex(SgdMapError, "ambiguous exact systematic ORF"):
                normalize_sgd_snapshot(raw, root / "output", MapBounds(), file_specs=specs)

    def test_record_and_line_bounds_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, specs = self._write_snapshot(root)
            with self.assertRaisesRegex(SgdMapError, "maxExternalRecords"):
                normalize_sgd_snapshot(
                    raw,
                    root / "output",
                    MapBounds(max_external_records=1),
                    file_specs=specs,
                )

    def test_materialized_dataset_resolver_accepts_only_literal_copy_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "stages" / "map" / "inputs" / "rawSgdMapping" / "sgd-map-raw"
            payload.mkdir(parents=True)
            base = {
                "resource": (
                    "omf://abiome/slp/datasetsnapshot/sgd-map-raw@sha256:" + "a" * 64
                ),
                "mode": "copy",
                "path": str(payload),
                "manifestDigest": "sha256:" + "b" * 64,
            }
            resolved = resolve_pinned_dataset_input(base, "rawSgdMapping")
            self.assertEqual(resolved.path, str(payload.resolve()))
            cases = (
                ({"path": str(payload)}, "spoofed materialized"),
                ({**base, "mode": "mount"}, "immutable copied"),
                ({**base, "manifestDigest": "latest"}, "admission-pinned"),
                (
                    {**base, "resource": base["resource"].replace("datasetsnapshot", "artifact")},
                    "kind must be DatasetSnapshot",
                ),
                (
                    {**base, "resource": base["resource"].split("@")[0] + "@latest"},
                    "admission-pinned",
                ),
            )
            for value, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(SgdMapError, message):
                    resolve_pinned_dataset_input(value, "rawSgdMapping")

    def test_symlinked_snapshot_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, specs = self._write_snapshot(root)
            target = raw / "SGD_features.tab"
            link = raw / "SGD_features-link.tab"
            try:
                os.symlink(target, link)
            except OSError:
                self.skipTest("symlink creation is not permitted on this Windows host")
            with self.assertRaisesRegex(SgdMapError, "must not be a symlink"):
                normalize_sgd_snapshot(raw, root / "output", MapBounds(), file_specs=specs)


if __name__ == "__main__":
    unittest.main()
