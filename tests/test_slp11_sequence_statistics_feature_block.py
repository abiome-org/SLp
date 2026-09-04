from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import struct
import sys
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "modules" / "slp-1-1-sequence-statistics-feature-block-v1"
SPEC = importlib.util.spec_from_file_location("slp11_sequence_feature_block", MODULE_ROOT / "feature_block.py")
assert SPEC and SPEC.loader
fb = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fb
SPEC.loader.exec_module(fb)


def cjson(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def pjson(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def jsonl(records: list[dict[str, object]]) -> bytes:
    return b"".join(cjson(record) for record in records)


def file_ref(path: str, payload: bytes, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {"path": path, "sha256": sha(payload), "bytes": len(payload)}
    if records is not None:
        result["records"] = records
    return result


def ustar(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            payload = members[name]
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


class Fixture:
    def __init__(self, root: Path, *, current_peptides: dict[str, str] | None = None) -> None:
        self.root = root
        self.inputs = root / "run" / "stages" / "build" / "inputs"
        self.current_peptides = current_peptides or {
            "SGD:S000000001": "MAC*",
            "SGD:S000000002": "MAC*",
            "SGD:S000000003": "MYYY*",
        }
        self.current_systematics = {
            "SGD:S000000001": "YAA001W",
            "SGD:S000000002": "YAA002W",
            "SGD:S000000003": "YAA003W",
        }
        self.universe_resource = "omf://test/slp/datasetsnapshot/test-universe@sha256:" + "1" * 64
        self.sequence_resource = "omf://test/slp/datasetsnapshot/test-sequences@sha256:" + "2" * 64
        self._make_universe()
        self._make_mapping()
        self._make_sequences()
        self.expected = fb.ExpectedContract(
            universe=fb.ExpectedDataset(
                self.universe_resource,
                "sha256:" + "3" * 64,
                fb._omf_tree_digest(tuple(
                    fb.FileSpec(name, len(payload), sha(payload))
                    for name, payload in sorted(self.universe_files.items())
                )),
                tuple(
                    fb.FileSpec(name, len(payload), sha(payload))
                    for name, payload in sorted(self.universe_files.items())
                ),
            ),
            sequences=fb.ExpectedDataset(
                self.sequence_resource,
                "sha256:" + "5" * 64,
                fb._omf_tree_digest(tuple(
                    fb.FileSpec(name, len(payload), sha(payload))
                    for name, payload in sorted(self.sequence_files.items())
                )),
                tuple(
                    fb.FileSpec(name, len(payload), sha(payload))
                    for name, payload in sorted(self.sequence_files.items())
                ),
            ),
            current_orfs=fb.ExpectedArtifact("sha256:" + "7" * 64, len(self.current_bytes), sha(self.current_bytes)),
            mapping_manifest=fb.ExpectedArtifact("sha256:" + "8" * 64, len(self.mapping_bytes), sha(self.mapping_bytes)),
            mapping_id="test-map:v1",
            mapping_sha256=self.mapping_sha,
            fasta_decompressed_bytes=len(self.fasta_bytes),
            fasta_decompressed_sha256=sha(self.fasta_bytes),
            fasta_records=5,
            current_orf_records=3,
            non_current_records=2,
            universe_genes=2,
            universe_proteins=1,
            universe_rows=3,
            current_outside_universe=1,
            present_values=63,
            universe_entity_key_sha256=self.entity_key_sha,
            universe_entity_jsonl_sha256=sha(self.entity_bytes),
            universe_manifest_sha256=sha(self.universe_manifest_bytes),
            universe_relation_jsonl_sha256=sha(self.relation_bytes),
            source_class_counts=(("verified-orf", 5), ("uncharacterized-orf", 0), ("dubious-orf", 0), ("transposable-element-gene", 0), ("pseudogene", 0), ("blocked-reading-frame", 0)),
            stop_absent_non_current=(("SGD:S000000005", "YAA005W"),),
            internal_stop_non_current=("SGD:S000000004",),
            non_m_start_non_current=("SGD:S000000005",),
            multi_target_peptide_sha256=(("UniProtKB:P00001", sha(self.current_peptides["SGD:S000000001"][:-1].encode())),),
        )
        self.universe = self._dataset_input("staticEntityUniverse", "test-universe", self.universe_resource, "sha256:" + "3" * 64, self.universe_files)
        self.sequences = self._dataset_input("sgdProteinSequences", "test-sequences", self.sequence_resource, "sha256:" + "5" * 64, self.sequence_files)
        self.current = self._artifact_input("sgdCurrentOrfs", "sha256:" + "7" * 64, self.current_bytes)
        self.mapping = self._artifact_input("sgdMappingManifest", "sha256:" + "8" * 64, self.mapping_bytes)

    def _make_universe(self) -> None:
        entities = [
            {"schema": fb.ENTITY_SCHEMA, "ncbiTaxon": 4932, "entityId": "SGD:S000000001", "entityClass": "gene", "usages": ["action", "relation-support"]},
            {"schema": fb.ENTITY_SCHEMA, "ncbiTaxon": 4932, "entityId": "SGD:S000000002", "entityClass": "gene", "usages": ["relation-support"]},
            {"schema": fb.ENTITY_SCHEMA, "ncbiTaxon": 4932, "entityId": "UniProtKB:P00001", "entityClass": "protein", "usages": ["readout-query"]},
        ]
        relations = [{
            "schema": fb.RELATION_SCHEMA,
            "proteinId": "UniProtKB:P00001",
            "sourceAccession": "P00001",
            "sourceAccessionType": {"source": "UniProtKB", "type": "UniProtKB ID", "namespaceInferred": False, "caseNormalization": "none"},
            "ncbiTaxon": 4932,
            "currentOrfRelations": ["SGD:S000000001", "SGD:S000000002"],
            "currentOrfRelationCount": 2,
            "chooseFirstAllowed": False,
        }]
        self.entity_bytes = jsonl(entities)
        self.relation_bytes = jsonl(relations)
        self.entity_key_sha = fb.framed_key_sha256((item["ncbiTaxon"], item["entityId"]) for item in entities)
        manifest = {
            "schema": fb.UNIVERSE_SCHEMA,
            "version": 1,
            "identityKey": ["ncbiTaxon", "entityId"],
            "ordering": "ascending-ncbiTaxon-then-codepoint-entityId",
            "source": {"id": "fixture", "release": "fixture-v1", "ncbiTaxon": 4932},
            "identityMapping": {"id": "test-map:v1", "sha256": "0" * 64},
            "semanticSetHashes": {"fullEntityKeySet": {"basis": "ncbiTaxon-TAB-entityId", "sha256": self.entity_key_sha}},
            "inputs": {},
            "entities": {"format": fb.ENTITY_SCHEMA, "file": file_ref("entities.jsonl", self.entity_bytes, 3), "counts": {}},
            "relations": {"format": fb.RELATION_SCHEMA, "file": file_ref("relations.jsonl", self.relation_bytes, 1), "relationSetSha256": sha(self.relation_bytes), "edges": 2, "oneToManyRecords": 1, "chooseFirstAllowed": False, "targetGenes": 2, "targetsInUniverse": 2},
            "contentPolicy": {"containsDisplaySymbols": False, "containsNumericFeatures": False, "containsOutcomesOrLabels": False, "containsTrainingPartitionAssignments": False, "crossTaxonIdentityMerge": False},
        }
        # Mapping SHA is replaced after the mapping fixture is constructed.
        self._universe_manifest = manifest

    def _make_mapping(self) -> None:
        current = []
        for curie, systematic in sorted(self.current_systematics.items()):
            current.append({
                "schema": fb.CURRENT_ORF_SCHEMA,
                "canonicalSgdCurie": curie,
                "systematicName": systematic,
                "featureQualifier": "Verified",
                "ncbiTaxon": 4932,
                "displayMetadata": {"aliases": [], "resolvesIdentity": False, "standardGeneName": None},
                "secondaryIdentifiers": [],
                "secondaryIdentifiersResolve": False,
            })
        self.current_bytes = jsonl(current)
        digest_basis = {
            "identityMappingId": "test-map:v1",
            "ncbiTaxon": 4932,
            "outputFiles": [{
                "name": "current-orfs.jsonl",
                "sha256": sha(self.current_bytes),
                "bytes": len(self.current_bytes),
                "records": 3,
            }],
        }
        self.mapping_sha = sha(cjson(digest_basis))
        self._universe_manifest["identityMapping"]["sha256"] = self.mapping_sha
        self.universe_manifest_bytes = pjson(self._universe_manifest)
        members = {
            "static-entity-universe/entities.jsonl": self.entity_bytes,
            "static-entity-universe/manifest.json": self.universe_manifest_bytes,
            "static-entity-universe/relations.jsonl": self.relation_bytes,
        }
        universe_tar = ustar(members)
        universe_audit = pjson({
            "schema": "slp.static-entity-universe-audit/v1",
            "inputs": {},
            "source": self._universe_manifest["source"],
            "identityMapping": self._universe_manifest["identityMapping"],
            "semanticSetHashes": self._universe_manifest["semanticSetHashes"],
            "outputs": {"archiveSha256": sha(universe_tar), "manifestSha256": sha(self.universe_manifest_bytes)},
        })
        self.universe_files = {"entity-universe-audit.json": universe_audit, "entity-universe.tar": universe_tar}
        mapping = {
            "schema": fb.MAPPING_MANIFEST_SCHEMA,
            "identityMappingId": "test-map:v1",
            "identityMappingSha256": self.mapping_sha,
            "ncbiTaxon": 4932,
            "digestBasis": digest_basis,
        }
        self.mapping_bytes = pjson(mapping)

    def _make_sequences(self) -> None:
        records = [(curie, self.current_systematics[curie], peptide) for curie, peptide in sorted(self.current_peptides.items())]
        records += [
            ("SGD:S000000004", "YAA004W", "M*A*"),
            ("SGD:S000000005", "YAA005W", "ACD"),
        ]
        chunks = []
        for curie, systematic, peptide in records:
            chunks.append(
                f">{systematic} GENE SGDID:{curie.removeprefix('SGD:')}, "
                f"Chr I from 1-2, Genome Release 64-5-1, Verified ORF, fixture\n{peptide}\n"
            )
        self.fasta_bytes = "".join(chunks).encode("ascii")
        gz = gzip.compress(self.fasta_bytes, mtime=0)
        self.sequence_files = {
            "dates_of_genome_releases.tab": b"R64-5-1\t2024-05-29\n",
            "orf_protein.README": b"fixture\n",
            "orf_trans_all_R64-5-1_20240529.fasta.gz": gz,
        }

    def _dataset_input(self, input_name: str, resource_name: str, resource: str, manifest: str, files: dict[str, bytes]):
        path = self.inputs / input_name / resource_name
        path.mkdir(parents=True)
        for name, payload in files.items():
            (path / name).write_bytes(payload)
        value = {"resource": resource, "mode": "copy", "path": str(path), "manifestDigest": manifest}
        return fb.resolve_pinned_dataset(value, input_name)

    def _artifact_input(self, input_name: str, digest: str, payload: bytes):
        path = self.inputs / input_name / "payload" / "payload"
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        value = {"resource": f"artifact:{digest}", "kind": "artifact", "artifacts": {"payload": digest}, "paths": {"payload": str(path)}, "path": str(path)}
        return fb.resolve_literal_artifact(value, input_name)

    def build(self, destination: Path | None = None):
        destination = destination or self.root / "output"
        result = fb.build_sequence_feature_block(
            self.universe, self.sequences, self.current, self.mapping,
            destination, fb.Bounds(max_fasta_bytes=1024 * 1024), expected=self.expected,
        )
        return destination, result


class SequenceStatisticsFeatureBlockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_is_deterministic_and_transform_is_exact(self) -> None:
        fixture = Fixture(self.root / "a")
        first, result = fixture.build()
        fixture2 = Fixture(self.root / "b")
        second, result2 = fixture2.build()
        archive = first / "sequence-feature-block.tar"
        self.assertEqual(archive.read_bytes(), (second / "sequence-feature-block.tar").read_bytes())
        self.assertEqual(result["archiveSha256"], result2["archiveSha256"])
        manifest = fb.validate_archive(archive, fb.Bounds(max_fasta_bytes=1024 * 1024), expected=fixture.expected)
        audit = fb.validate_audit(first / "sequence-feature-block-audit.json", archive, fb.Bounds(max_fasta_bytes=1024 * 1024), expected=fixture.expected)
        self.assertEqual(manifest["counts"]["rows"], 3)
        self.assertEqual(audit["counts"]["presentValues"], 63)
        with tarfile.open(archive, "r:") as handle:
            values = handle.extractfile("static-feature-block/values.npy").read()
            present = handle.extractfile("static-feature-block/present.npy").read()
            provenance = [json.loads(line) for line in handle.extractfile("static-feature-block/sequence-provenance.jsonl")]
        values_data = fb._parse_npy(values, "<f4", (3, 21), "values")
        rows = [struct.unpack("<21f", values_data[index:index + 84]) for index in range(0, len(values_data), 84)]
        self.assertEqual(rows[0][0], 3 / 4096)
        self.assertAlmostEqual(sum(rows[0][1:]), 1.0, places=6)
        self.assertEqual(fb._parse_npy(present, "|b1", (3, 21), "present"), b"\x01" * 63)
        self.assertEqual(provenance[0]["aminoAcidCounts"], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(provenance[2]["sourceSequenceIds"], ["SGD:S000000001", "SGD:S000000002"])

    def test_multi_target_peptides_must_be_identical(self) -> None:
        fixture = Fixture(self.root, current_peptides={
            "SGD:S000000001": "MAC*",
            "SGD:S000000002": "MAA*",
            "SGD:S000000003": "MYYY*",
        })
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "exact consensus"):
            fixture.build()

    def test_current_peptide_must_start_m_and_have_one_terminal_stop(self) -> None:
        for peptide in ("ACD*", "Mac*", "MAC", "MA*C*"):
            with self.subTest(peptide=peptide):
                root = self.root / hashlib.sha256(peptide.encode()).hexdigest()[:8]
                fixture = Fixture(root, current_peptides={
                    "SGD:S000000001": peptide,
                    "SGD:S000000002": peptide,
                    "SGD:S000000003": "MYYY*",
                })
                with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "current ORF|alphabet"):
                    fixture.build()

    def test_excluded_anomalies_are_preserved_not_featurized(self) -> None:
        fixture = Fixture(self.root)
        output, _ = fixture.build()
        with tarfile.open(output / "sequence-feature-block.tar", "r:") as handle:
            rows = [json.loads(line) for line in handle.extractfile("static-feature-block/excluded-non-current.jsonl")]
        self.assertEqual([row["sequenceId"] for row in rows], ["SGD:S000000004", "SGD:S000000005"])
        self.assertEqual(rows[0]["internalStopCount"], 1)
        self.assertFalse(rows[1]["terminalStopPresent"])
        self.assertFalse(rows[1]["startsWithMethionine"])
        self.assertNotIn("canonicalPeptideSha256", rows[0])

    def test_source_class_uses_structural_token_not_curated_free_text(self) -> None:
        cases = (
            (
                (
                    "Chr I from 2169-1807, Genome Release 64-5-1, reverse complement, "
                    'Verified ORF, "fixture"'
                ),
                "verified-orf",
            ),
            (
                (
                    "Chr I from 2480-2707, Genome Release 64-5-1, "
                    'Uncharacterized ORF, "fixture"'
                ),
                "uncharacterized-orf",
            ),
            (
                (
                    "Chr I from 227742-228953, Genome Release 64-5-1, Dubious ORF, "
                    '"Nonfunctional protein; blocked reading frame in curated prose"'
                ),
                "dubious-orf",
            ),
            (
                (
                    "Chr I from 164187-160597, Genome Release 64-5-1, reverse complement, "
                    'transposable_element_gene, "fixture"'
                ),
                "transposable-element-gene",
            ),
            (
                (
                    "Chr I from 218140-219145, Genome Release 64-5-1, "
                    'pseudogene, "fixture"'
                ),
                "pseudogene",
            ),
            (
                (
                    "Chr IV from 721481-721071, Genome Release 64-5-1, reverse complement, "
                    'blocked_reading_frame, "fixture"'
                ),
                "blocked-reading-frame",
            ),
        )
        for description, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(fb._source_class(description), expected)
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "structural class"):
            fb._source_class(
                'Chr I from 1-2, Genome Release 64-5-1, unknown_class, "Dubious ORF"'
            )

    def test_values_tamper_fails_even_when_manifest_hash_is_rewritten(self) -> None:
        fixture = Fixture(self.root)
        output, _ = fixture.build()
        archive = output / "sequence-feature-block.tar"
        with tarfile.open(archive, "r:") as handle:
            blobs = {member.name: handle.extractfile(member).read() for member in handle.getmembers()}
        values_name = "static-feature-block/values.npy"
        values = bytearray(blobs[values_name])
        values[-1] ^= 1
        blobs[values_name] = bytes(values)
        manifest_name = "static-feature-block/manifest.json"
        manifest = json.loads(blobs[manifest_name])
        manifest["files"]["values"] = file_ref("values.npy", blobs[values_name])
        blobs[manifest_name] = pjson(manifest)
        archive.write_bytes(ustar(blobs))
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "sufficient-statistics transform"):
            fb.validate_archive(archive, fb.Bounds(), expected=fixture.expected)

    def test_production_output_pins_reject_coordinated_provenance_tamper(self) -> None:
        fixture = Fixture(self.root)
        output, result = fixture.build()
        archive = output / "sequence-feature-block.tar"
        pinned = replace(
            fixture.expected,
            output_sequence_provenance_sha256=result["sequenceProvenanceSha256"],
            output_excluded_non_current_sha256=result["excludedNonCurrentSha256"],
        )
        fb.validate_archive(archive, fb.Bounds(), expected=pinned)
        with tarfile.open(archive, "r:") as handle:
            blobs = {member.name: handle.extractfile(member).read() for member in handle.getmembers()}
        provenance_name = "static-feature-block/sequence-provenance.jsonl"
        provenance = [json.loads(line) for line in blobs[provenance_name].splitlines()]
        provenance[0]["canonicalPeptideSha256"] = "f" * 64
        blobs[provenance_name] = jsonl(provenance)
        manifest_name = "static-feature-block/manifest.json"
        manifest = json.loads(blobs[manifest_name])
        manifest["files"]["sequenceProvenance"] = file_ref(
            "sequence-provenance.jsonl", blobs[provenance_name], len(provenance)
        )
        manifest["semanticHashes"]["sequenceProvenanceSha256"] = sha(blobs[provenance_name])
        blobs[manifest_name] = pjson(manifest)
        archive.write_bytes(ustar(blobs))
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "production-pinned source/relation map"):
            fb.validate_archive(archive, fb.Bounds(), expected=pinned)

    def test_production_output_pin_rejects_coordinated_exclusion_tamper(self) -> None:
        fixture = Fixture(self.root)
        output, result = fixture.build()
        archive = output / "sequence-feature-block.tar"
        pinned = replace(
            fixture.expected,
            output_sequence_provenance_sha256=result["sequenceProvenanceSha256"],
            output_excluded_non_current_sha256=result["excludedNonCurrentSha256"],
        )
        with tarfile.open(archive, "r:") as handle:
            blobs = {member.name: handle.extractfile(member).read() for member in handle.getmembers()}
        excluded_name = "static-feature-block/excluded-non-current.jsonl"
        excluded = [json.loads(line) for line in blobs[excluded_name].splitlines()]
        excluded[0]["rawSequenceSha256"] = "e" * 64
        blobs[excluded_name] = jsonl(excluded)
        manifest_name = "static-feature-block/manifest.json"
        manifest = json.loads(blobs[manifest_name])
        manifest["files"]["excludedNonCurrent"] = file_ref(
            "excluded-non-current.jsonl", blobs[excluded_name], len(excluded)
        )
        blobs[manifest_name] = pjson(manifest)
        archive.write_bytes(ustar(blobs))
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "production-pinned quarantine"):
            fb.validate_archive(archive, fb.Bounds(), expected=pinned)

    def test_excluded_cardinality_is_bound_to_manifest_count_without_output_pin(self) -> None:
        fixture = Fixture(self.root)
        output, _ = fixture.build()
        archive = output / "sequence-feature-block.tar"
        with tarfile.open(archive, "r:") as handle:
            blobs = {member.name: handle.extractfile(member).read() for member in handle.getmembers()}
        excluded_name = "static-feature-block/excluded-non-current.jsonl"
        excluded = [json.loads(line) for line in blobs[excluded_name].splitlines()]
        ordinary = dict(excluded[-1])
        ordinary.update(
            sequenceId="SGD:S000000006",
            systematicName="YAA006W",
            terminalStopPresent=True,
            internalStopCount=0,
            startsWithMethionine=True,
            rawSequenceSha256="d" * 64,
        )
        excluded.append(ordinary)
        excluded.sort(key=lambda row: row["sequenceId"])
        blobs[excluded_name] = jsonl(excluded)
        manifest_name = "static-feature-block/manifest.json"
        manifest = json.loads(blobs[manifest_name])
        manifest["files"]["excludedNonCurrent"] = file_ref(
            "excluded-non-current.jsonl", blobs[excluded_name], len(excluded)
        )
        blobs[manifest_name] = pjson(manifest)
        archive.write_bytes(ustar(blobs))
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "excluded sequence count"):
            fb.validate_archive(archive, fb.Bounds(), expected=fixture.expected)

    def test_json_boolean_fields_do_not_accept_integer_aliases(self) -> None:
        fixture = Fixture(self.root)
        output, _ = fixture.build()
        archive = output / "sequence-feature-block.tar"
        with tarfile.open(archive, "r:") as handle:
            blobs = {member.name: handle.extractfile(member).read() for member in handle.getmembers()}
        manifest_name = "static-feature-block/manifest.json"
        manifest = json.loads(blobs[manifest_name])
        manifest["featureDefinition"]["clipping"] = 0
        blobs[manifest_name] = pjson(manifest)
        archive.write_bytes(ustar(blobs))
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "immutable feature definition"):
            fb.validate_archive(archive, fb.Bounds(), expected=fixture.expected)

        fixture = Fixture(self.root / "audit")
        output, _ = fixture.build()
        audit_path = output / "sequence-feature-block-audit.json"
        audit = json.loads(audit_path.read_bytes())
        audit["accessBoundary"]["heldRosterConsumed"] = 0
        audit_path.write_bytes(pjson(audit))
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "boundary"):
            fb.validate_audit(
                audit_path,
                output / "sequence-feature-block.tar",
                fb.Bounds(),
                expected=fixture.expected,
            )

    def test_npy_header_and_presence_are_fail_closed(self) -> None:
        valid = fb._npy_bytes("<f4", (2, 21), b"\0" * 168)
        for payload in (
            valid + b"x",
            valid.replace(b"'<f4'", b"'>f4'", 1),
            fb._npy_bytes("|b1", (1, 21), b"\x01" * 20 + b"\0"),
        ):
            with self.subTest(size=len(payload)):
                if payload.startswith(b"\x93NUMPY") and b"|b1" in payload:
                    self.assertIn(0, fb._parse_npy(payload, "|b1", (1, 21), "mask"))
                else:
                    with self.assertRaises(fb.SequenceFeatureBlockError):
                        fb._parse_npy(payload, "<f4", (2, 21), "values")

    def test_audit_tamper_is_rejected(self) -> None:
        fixture = Fixture(self.root)
        output, _ = fixture.build()
        audit_path = output / "sequence-feature-block-audit.json"
        audit = json.loads(audit_path.read_bytes())
        audit["accessBoundary"]["heldRosterConsumed"] = True
        audit_path.write_bytes(pjson(audit))
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "boundary"):
            fb.validate_audit(audit_path, output / "sequence-feature-block.tar", fb.Bounds(), expected=fixture.expected)

    def test_dataset_and_artifact_shapes_reject_spoofing(self) -> None:
        fixture = Fixture(self.root)
        dataset_value = {"resource": fixture.universe.resource, "mode": "link", "path": str(fixture.universe.path), "manifestDigest": fixture.universe.manifest_digest}
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "copied"):
            fb.resolve_pinned_dataset(dataset_value, "staticEntityUniverse")
        artifact_value = {"resource": "artifact:sha256:" + "7" * 64, "kind": "artifact", "artifacts": {"payload": "sha256:" + "7" * 64}, "paths": {"payload": str(fixture.current.path)}, "path": str(fixture.current.path), "extra": True}
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "spoofed"):
            fb.resolve_literal_artifact(artifact_value, "sgdCurrentOrfs")

    def test_omf_tree_digest_is_reconstructed_from_the_pinned_file_set(self) -> None:
        for dataset in (
            fb.PRODUCTION_CONTRACT.universe,
            fb.PRODUCTION_CONTRACT.sequences,
        ):
            with self.subTest(resource=dataset.resource):
                self.assertEqual(fb._omf_tree_digest(dataset.files), dataset.tree_digest)
        fixture = Fixture(self.root)
        drifted = replace(fixture.expected.universe, tree_digest="sha256:" + "0" * 64)
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "tree digest"):
            fb._verify_dataset(fixture.universe, drifted)

    def test_destination_is_never_overwritten(self) -> None:
        fixture = Fixture(self.root)
        destination, _ = fixture.build()
        sentinel = (destination / "sequence-feature-block.tar").read_bytes()
        with self.assertRaisesRegex(fb.SequenceFeatureBlockError, "must not already exist"):
            fixture.build(destination)
        self.assertEqual((destination / "sequence-feature-block.tar").read_bytes(), sentinel)

    def test_module_is_self_contained_and_workload_has_only_four_inputs(self) -> None:
        module_text = (MODULE_ROOT / "module.yaml").read_text()
        workload_text = (ROOT / "workloads" / "slp-1-1-sequence-statistics-feature-block-v1.yaml.tmpl").read_text()
        self.assertIn("dependencyDigest: sha256:e3b0c442", module_text)
        self.assertIn("dataset/slp-1-1-static-entity-universe-v1", workload_text)
        self.assertIn("dataset/slp-1-1-sgd-protein-sequences-r64-5-1", workload_text)
        input_block = workload_text.split("        inputs:\n", 1)[1].split("        config:\n", 1)[0]
        self.assertEqual(sum(line.startswith("          ") and not line.startswith("            ") for line in input_block.splitlines()), 4)
        for forbidden in ("heldRoster:", "observation", "reward:", "labels:"):
            self.assertNotIn(forbidden, input_block)


if __name__ == "__main__":
    unittest.main()
