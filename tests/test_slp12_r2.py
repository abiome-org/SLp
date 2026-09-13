"""Numerical invariants and strict exposure contracts of the r2 replacement."""

import copy
import importlib.util
from pathlib import Path
import sys
import io
import json
import pickle
import numpy as np

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(
        "r2_" + name, ROOT / "modules/slp-1-2-r2" / (name + ".py")
    )
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


model = module("model")
records = module("records")
previous_records = sys.modules.get("records")
sys.modules["records"] = records
data = module("data")
disk = module("disk_corpus")
if previous_records is None:
    del sys.modules["records"]
else:
    sys.modules["records"] = previous_records
train = module("train")
evaluate = module("evaluate")
extract_features = module("extract_features")
previous_data = sys.modules.get("data")
sys.modules["data"] = data
packed = module("packed")
if previous_data is None:
    del sys.modules["data"]
else:
    sys.modules["data"] = previous_data


def prep_module(name):
    previous = list(sys.path)
    sys.path.insert(0, str(ROOT / "storage/prep"))
    try:
        spec = importlib.util.spec_from_file_location(
            "prep_" + name, ROOT / "storage/prep" / (name + ".py")
        )
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    finally:
        sys.path[:] = previous


def test_packed_cloud_roundtrip_exclusions_sampling_and_actions(tmp_path, monkeypatch):
    pack = prep_module("pack_corpus")
    monkeypatch.setenv("SLP_JOB", "synthetic-fixture")

    def put(path, *, data=None, prep=False):
        (tmp_path / Path(path).name).write_bytes(data)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(pack, "request", put)
    monkeypatch.setattr(
        pack,
        "save",
        lambda name, value: (tmp_path / (name + ".json")).write_text(
            pack.encode(value)
        ),
    )
    genes = ["9606:A", "9606:B", "9606:C", "559292:A"]
    writer = pack.Writer(genes, "fixture")
    template = dict(
        source="fixture",
        study="fixture",
        taxon=9606,
        context="cell",
        assay="fitness",
        kind="fitness",
        units="score",
        scope="cell",
        license="synthetic",
        training_allowed=True,
        lineage={"fixture": True},
    )
    writer.add(
        template=template, genes=["9606:A"], value=1e9, native_row=0, unit="held"
    )
    for i, value in enumerate((1.0, 3.0)):
        writer.add(
            template=template,
            genes=["9606:C", "9606:B"],
            value=value,
            native_row=i + 1,
            unit="rep" + str(i),
            mechanisms=["crispri", "knockout"],
            method=["dCas9-KRAB", "unknown"],
            hours=[336.0, None],
        )
    writer.add(
        template={**template, "taxon": 559292},
        genes=["559292:A"],
        value=2.0,
        native_row=3,
        unit="yeast",
        query="559292:A",
    )
    writer.add(
        template={**template, "training_allowed": False},
        genes=["9606:C"],
        value=1e9,
        native_row=4,
        unit="quarantine",
    )
    writer.finish()
    cache = packed.materialize(
        [tmp_path / "fixture-manifest.json"], tmp_path / "cache", genes
    )
    fold = records.Fold("strict", frozenset({"9606:A"}))
    view = packed.View(cache, genes, fold, "pretrain")
    assert view.receipt["fitting_rows"] == 3
    assert len(view.receipt["quarantined_templates"]) == 1
    assert view.scales[0, 0] == pytest.approx(2.0)
    sampler = packed.Sampler(view, {"fitness": 1.0}, max_cycles=2)
    sampler.draw()
    state = copy.deepcopy(sampler.state_dict())
    expected = sampler.draw()
    sampler.load_state_dict(state)
    np.testing.assert_array_equal(sampler.draw(), expected)
    human = view.rows[[1]]
    assert human["targets"].tolist() == [[1, 2, -1, -1]]
    assert human["method"][0, :2].tolist() == [
        pack.METHODS["unknown"],
        pack.METHODS["dCas9-KRAB"],
    ]
    assert not human["known"][0, 0, 1] and human["known"][0, 1, 1]
    features = type(
        "Features",
        (),
        dict(
            index={g: i for i, g in enumerate(genes)},
            sequence=np.ones((4, 12), np.float32),
            annotation=np.ones((4, 4), np.float32),
            known=np.ones((4, 2), bool),
        ),
    )()
    config = model.Config(
        width=32, layers=2, heads=4, sequence_dim=12, annotation_dim=4, context_dim=8
    )
    builder = packed.Builder(
        features, view.templates, view.scales, packed.vocabulary(view.templates), config
    )
    batch = builder([human])
    loss = train.objective(model.WorldModel(config)(batch), batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert packed.View(cache, genes, fold, "adapt").receipt["fitting_rows"] == 2


def test_pinned_split_data_reader_handles_numpy_torch_and_blocks_code():
    safe = prep_module("safe_arrays")
    expected = {
        "array": np.arange(6).reshape(2, 3),
        "tensor": torch.tensor([[1, 2], [3, 4]]),
    }
    value = safe.load(pickle.dumps(expected))
    np.testing.assert_array_equal(value["array"], expected["array"])
    np.testing.assert_array_equal(value["tensor"], expected["tensor"].numpy())

    class Executable:
        def __reduce__(self):
            return (eval, ("1+1",))

    with pytest.raises(ValueError, match="Unapproved pickle constructor"):
        safe.load(pickle.dumps(Executable()))


def test_benchmark_inner_access_and_portable_artifact(tmp_path, monkeypatch):
    import gzip
    import hashlib

    monkeypatch.setitem(sys.modules, "records", records)
    monkeypatch.setitem(sys.modules, "data", data)
    monkeypatch.setitem(sys.modules, "model", model)
    benchmark_module = module("benchmarks")
    artifact = module("artifact")
    root = tmp_path / "features"
    root.mkdir()
    genes = ["9606:" + g for g in "ABCDEF"]
    (root / "genes.json").write_text(json.dumps(genes))
    for name, value in [
        ("sequence.npy", np.ones((6, 12), np.float32)),
        ("annotation.npy", np.ones((6, 4), np.float32)),
        ("known.npy", np.ones((6, 2), bool)),
    ]:
        np.save(root / name, value, allow_pickle=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "slp.static-features/v2",
                "files": {p.name: data.digest(p) for p in root.iterdir()},
                "fitted_human_intervention_genes": [],
            }
        )
    )
    features = data.Features(root)
    directory = tmp_path / "benchmark"
    directory.mkdir()
    rows = [
        {
            "row": i,
            "targets": ["9606:" + a, "9606:" + b],
            "label": y,
            "context": "pan-cancer",
        }
        for i, (a, b, y) in enumerate(
            [("E", "F", 0), ("E", "F", 1), ("C", "D", 1), ("C", "E", 0), ("A", "E", 1)]
        )
    ]
    payload = gzip.compress(b"\n".join(json.dumps(r).encode() for r in rows))
    (directory / "train.jsonl.gz").write_bytes(payload)
    (directory / "train.json").write_text(
        json.dumps(
            {
                "rows": len(rows),
                "shards": [
                    {
                        "name": "train.jsonl.gz",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        )
    )
    (directory / "fold.json").write_text(
        json.dumps(
            {
                "name": "fixture",
                "benchmark": "fixture",
                "protocol": {},
                "outer_held": genes[:2],
                "inner_held": genes[2:4],
                "partitions": {"train": "train.json", "test": "not-downloaded.json"},
            }
        )
    )
    benchmark = benchmark_module.Benchmark(directory / "fold.json", features)
    fitting = benchmark.records("inner", "fit")
    assert [r.value for r in fitting] == [
        0.0,
        1.0,
    ]  # Published conflicting rows survive.
    assert len(benchmark.labels("inner", "evaluate")) == 1
    assert len(benchmark.labels("outer", "fit")) == 4
    benchmark_module.verify_training_labels(fitting, benchmark.fold("inner"))
    config = model.Config(
        width=32, layers=2, heads=4, sequence_dim=12, annotation_dim=4, context_dim=8
    )
    net = model.WorldModel(config).eval()
    vocabulary = {
        "assay": {"human-SL": 0},
        "taxon": {"9606": 0},
        "scope": {"pan-cancer": 0},
        "mechanism": {"unknown": 0},
        "method": {"unknown": 0},
    }
    builder = data.BatchBuilder(features, data.Scales([]), vocabulary, config)
    batch = builder([[r] for r in fitting])
    with torch.no_grad():
        expected = net(batch)["sl_logit"]
    artifact.export(
        tmp_path / "bundle",
        net,
        root,
        vocabulary=vocabulary,
        basal={},
        provenance={"fixture": True},
    )
    restored, *_ = artifact.load(tmp_path / "bundle")
    with torch.no_grad():
        torch.testing.assert_close(
            restored(batch)["sl_logit"], expected, rtol=0, atol=0
        )
    with (tmp_path / "bundle/weights.pt").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        artifact.load(tmp_path / "bundle")


def test_dense_populations_filter_before_scaling_and_resume_mixture(
    tmp_path, monkeypatch
):
    import hashlib

    monkeypatch.setitem(sys.modules, "data", data)
    monkeypatch.setitem(sys.modules, "packed", packed)
    monkeypatch.setitem(sys.modules, "cloud_io", module("cloud_io"))
    population = module("population")
    prep = prep_module("prepare_populations")
    genes = ["9606:A", "9606:B", "9606:C"]
    units = prep.Units(genes, "fixture")
    template = {
        "source": "rna-fixture",
        "study": "fixture",
        "taxon": 9606,
        "context": "control",
        "assay": "rna-fixture",
        "kind": "rna",
        "units": "measured-score",
        "scope": "population",
        "training_allowed": True,
        "license": "synthetic",
        "lineage": {"fixture": True},
    }
    for i, g in enumerate(genes):
        units.add(template=template, genes=[g], value=0.0, native_row=i, unit=i)
    arrays = {
        "units.npy": units.array(),
        "query-indices.npy": np.arange(3, dtype=np.int32),
        "targets.npy": np.array(
            [[1e9] * 3, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32
        ),
        "observed.npy": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], bool),
    }
    for name, value in arrays.items():
        np.save(tmp_path / name, value, allow_pickle=False)
    manifest = {
        "schema": "slp.population-pack/v1",
        "outcome_fitted_transforms": [],
        "files": {name: {"sha256": data.digest(tmp_path / name)} for name in arrays},
        "genes_sha256": hashlib.sha256(packed.encode(genes).encode()).hexdigest(),
        "units": 3,
        "queries": 3,
        "templates": units.templates,
    }
    path = tmp_path / "population-manifest.json"
    path.write_text(json.dumps(manifest))
    view = population.View(
        path, genes, records.Fold("fixture", frozenset({"9606:A"})), "pretrain"
    )
    assert view.receipt["fitting_units"] == 2
    assert view.scales[0, 0] == pytest.approx(3.0)
    mixture = population.Mixture([view], {"rna": 1.0}, max_cycles=1)
    state = copy.deepcopy(mixture.state_dict())
    expected = mixture.draw()
    mixture.load_state_dict(state)
    np.testing.assert_array_equal(mixture.draw(), expected)
    assert (
        0 in expected["query"]
    )  # A held intervention can still be a measured coordinate.
    assert not (expected["targets"] == 0).any()
    mixture.draw()
    with pytest.raises(StopIteration):
        mixture.draw()


def fixture():
    torch.manual_seed(29)
    config = model.Config(
        width=32, layers=2, heads=4, sequence_dim=12, annotation_dim=4, context_dim=8
    )
    net = model.WorldModel(config).eval()
    b = {
        "context": torch.randn(2, 8),
        "context_known": torch.tensor([True, False]),
        "assay": torch.zeros(2, dtype=torch.long),
        "taxon": torch.zeros(2, dtype=torch.long),
        "scope": torch.zeros(2, dtype=torch.long),
    }
    for prefix, n in (("observation", 3), ("action", 2), ("query", 4)):
        b[prefix + "_sequence"] = torch.randn(2, n, 12)
        b[prefix + "_annotation"] = torch.randn(2, n, 4)
        b[prefix + "_known"] = torch.ones(2, n, 2, dtype=torch.bool)
        b[prefix + "_mask"] = torch.ones(2, n, dtype=torch.bool)
    b["observation_values"] = torch.randn(2, 3)
    b["observation_kind"] = torch.zeros(2, 3, dtype=torch.long)
    b["observation_mask"][1, 2] = False
    b["action_values"] = torch.randn(2, 2, 3)
    b["action_numeric_known"] = torch.ones(2, 2, 3, dtype=torch.bool)
    b["action_mechanism"] = torch.ones(2, 2, dtype=torch.long)
    b["action_method"] = torch.ones(2, 2, dtype=torch.long)
    b["query_kind"] = torch.tensor([[0, 2, 3, 4], [1, 2, 3, 4]])
    return net, b


def permute(b, prefix, indices):
    return {k: v[:, indices] if k.startswith(prefix + "_") else v for k, v in b.items()}


def test_unordered_actions_and_independent_queries():
    net, b = fixture()
    with torch.no_grad():
        expected = net(b)["sl_logit"]
        torch.testing.assert_close(
            net(permute(b, "action", [1, 0]))["sl_logit"], expected
        )
        torch.testing.assert_close(
            net(permute(b, "query", [3]))["sl_logit"], expected[:, [3]]
        )
        changed = copy.deepcopy(b)
        changed["query_sequence"][:, :3] *= 300
        torch.testing.assert_close(net(changed)["sl_logit"][:, 3], expected[:, 3])


def test_cached_context_matches_and_rejects_stale_inputs_or_weights():
    net, b = fixture()
    with torch.no_grad():
        cache = net.prepare_context(b)
        torch.testing.assert_close(net(b, cache)["location"], net(b)["location"])
        changed = copy.deepcopy(b)
        changed["context"][0, 0] += 1
        with pytest.raises(ValueError, match="inputs changed"):
            net(changed, cache)
        next(net.parameters()).add_(0.01)
        with pytest.raises(ValueError, match="different model state"):
            net(b, cache)


def test_missing_values_and_padding_do_not_change_prediction():
    net, b = fixture()
    b["action_known"][..., 0] = False
    b["action_numeric_known"][:] = False
    with torch.no_grad():
        expected = net(b)["location"]
        b["action_sequence"][:] = float("nan")
        b["action_values"][:] = float("nan")
        b["observation_values"][1, 2] = 100000
        b["observation_sequence"][1, 2] = float("nan")
        b["context"][1] = float("nan")
        torch.testing.assert_close(net(b)["location"], expected)


def test_sl_gradients_reach_trunk_without_gene_table():
    net, b = fixture()
    net.train()
    logits = net(b)["sl_logit"][:, 3]
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 1.0])
    ).backward()
    assert net.blocks[0].qkv.weight.grad.abs().sum() > 0
    assert net.identity[0].weight.grad.abs().sum() > 0
    assert not any(
        name.split(".")[0] in {"entity", "gene_id"}
        for name, _ in net.named_parameters()
    )


def row(name, targets, taxon=9606, **kwargs):
    return records.Record(
        record_id=name,
        source_id="source",
        study_id="study",
        experiment_id=name,
        replicate_id="r1",
        taxon=taxon,
        targets=tuple(sorted(f"{taxon}:{g}" for g in targets)),
        context="cell",
        assay="screen",
        kind="fitness",
        units="logfitness",
        value=-0.4,
        query_gene=None,
        mechanism="knockout",
        method="cas9",
        label_scope="cell",
        role="quantitative",
        license="CC BY 4.0",
        raw_object="sha256:raw",
        raw_locator="row:1",
        **kwargs,
    )


def test_cv3_excludes_every_human_intervention_but_not_output_or_nonhuman():
    fold = records.Fold("outer1-inner1", frozenset({"9606:A"}), frozenset({"9606:B"}))
    corpus = [
        row("outer", ["A", "C"]),
        row("inner", ["B"]),
        row("fit", ["C"]),
        row("yeast", ["A"], 559292),
    ]
    selected, receipt = records.fitting_view(corpus, fold, "pretrain")
    assert [r.record_id for r in selected] == ["fit", "yeast"]
    assert receipt["human_intervention_genes"] == ["9606:C"]
    adapted, _ = records.fitting_view(corpus, fold, "adapt")
    assert [r.record_id for r in adapted] == ["fit"]
    output = records.Record.from_dict(
        {**row("output", ["C"]).__dict__, "query_gene": "9606:A"}
    )
    assert fold.permits(output, "pretrain")


def test_derived_data_cannot_hide_forbidden_parent():
    fold = records.Fold("test", frozenset({"9606:A"}))
    with pytest.raises(ValueError, match="lineage"):
        records.fitting_view(
            [row("held", ["A"]), row("derived", ["C"], lineage=("held",))],
            fold,
            "pretrain",
        )


def test_exact_benchmark_resolution_and_conflicting_duplicates():
    with pytest.raises(ValueError, match="Unresolved"):
        records.resolve_benchmark([{"a": "A", "b": "missing"}], {"A": "9606:1"})
    first = row("one", ["A", "B"])
    second = records.Record.from_dict(
        {**first.__dict__, "record_id": "copy", "source_id": "other"}
    )
    kept, aliases = records.deduplicate([first, second])
    assert kept == [first] and aliases == {"copy": "one"}
    conflict = records.Record.from_dict({**second.__dict__, "value": 0.2})
    with pytest.raises(ValueError, match="Conflicting"):
        records.deduplicate([first, conflict])


def test_sampler_cycles_distinct_conditions_and_restores_exactly():
    rows = [row(str(i), [str(i)]) for i in range(6)]
    sampler = data.ConditionSampler(rows, {"fitness": 1.0}, seed=93, max_cycles=2)
    first_cycle = [sampler.draw()[0].record_id for _ in rows]
    assert len(set(first_cycle)) == 6
    state = copy.deepcopy(sampler.state_dict())
    expected = [sampler.draw()[0].record_id for _ in rows]
    with pytest.raises(StopIteration):
        sampler.draw()
    restored = data.ConditionSampler(rows, {"fitness": 1.0}, seed=100, max_cycles=2)
    restored.load_state_dict(state)
    assert [restored.draw()[0].record_id for _ in rows] == expected


def test_scales_are_fitted_after_exclusions():
    ordinary = row("fit", ["C"])
    extreme = records.Record.from_dict({**row("held", ["A"]).__dict__, "value": 1e9})
    corpus = data.Corpus(
        [ordinary, extreme], records.Fold("x", frozenset({"9606:A"})), "pretrain"
    )
    assert corpus.scales.transform(ordinary) == 0.0


def test_checkpoint_replays_optimizer_rng_and_sampler(tmp_path):
    net, b = fixture()
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    sampler = data.ConditionSampler(
        [row(str(i), [str(i)]) for i in range(6)], {"fitness": 1.0}
    )
    b["target"] = torch.randn(2, 4)
    b["target"][:, 3] = torch.tensor([0.0, 1.0])

    def update():
        optimizer.zero_grad(set_to_none=True)
        loss = train.objective(net(b), b)
        loss.backward()
        optimizer.step()

    update()
    identity = {"fold": "fixture", "data_sha256": "synthetic", "total_updates": 2}
    path = tmp_path / "checkpoint.pt"
    train.save_checkpoint(path, net, optimizer, sampler, update=1, identity=identity)
    draw = sampler.draw()[0].record_id
    expected_random = torch.rand(4)
    update()
    expected = copy.deepcopy(net.state_dict())
    assert (
        train.resume(path, net, optimizer, sampler, identity=identity, device="cpu")
        == 1
    )
    assert sampler.draw()[0].record_id == draw
    torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
    update()
    for key, value in net.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity mismatch"):
        train.resume(
            path, net, optimizer, sampler, identity={"fold": "other"}, device="cpu"
        )


def test_metrics_ties_and_selector_rejects_outer_or_partial_evidence():
    tied = evaluate.binary_metrics([0, 1, 0, 1], [0, 0, 0, 0])
    assert tied["average_precision"] == 0.5
    assert tied["auroc"] == 0.5
    reports = [
        {
            "partition": "inner",
            "forbidden_human_exposures": 0,
            "benchmark": "a",
            "fold": "i1",
            "checkpoint": "x",
            "average_precision": 0.8,
        }
    ]
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate.select_checkpoint(reports, ["a", "b"], ["i1"])
    assert evaluate.select_checkpoint(reports, ["a"], ["i1"])["selected"] == "x"
    reports[0]["partition"] = "outer"
    with pytest.raises(ValueError, match="inner-CV3"):
        evaluate.select_checkpoint(reports, ["a"], ["i1"])


def test_disk_view_enforces_fold_and_sampler_restores(tmp_path):
    path = tmp_path / "corpus.sqlite"
    rows = [row(str(i), [str(i)]) for i in range(5)] + [
        row("held", ["A"]),
        row("yeast", ["A"], taxon=559292),
    ]
    disk.build(path, rows, {"fixture": "synthetic"})
    fold = records.Fold("cold", frozenset({"9606:A"}))
    view = disk.View(path, fold, "pretrain")
    try:
        assert view.receipt["forbidden_human_exposures"] == 0
        sampler = disk.Sampler(view, {"fitness": 1.0}, seed=8, max_cycles=2)
        first = [sampler.draw()[0].record_id for _ in range(6)]
        assert len(set(first)) == 6 and "held" not in first
        state = copy.deepcopy(sampler.state_dict())
        expected = [sampler.draw()[0].record_id for _ in range(6)]
        sampler.load_state_dict(state)
        assert [sampler.draw()[0].record_id for _ in range(6)] == expected
        with pytest.raises(StopIteration):
            sampler.draw()
    finally:
        view.close()


def test_long_protein_pooling_counts_every_residue_once():
    import numpy as np

    for length in (1, 1022, 1023, 3000, 35000):
        pieces = extract_features.weighted_windows("A" * length)
        assert all(len(fragment) <= 1022 for fragment, _ in pieces)
        assert np.isclose(sum(weights.sum() for _, weights in pieces), length)


def test_ordinary_fit_pretrain_then_human_adaptation_and_initializer_exclusion(
    tmp_path, monkeypatch
):
    import subprocess
    import time
    import gzip

    pack = prep_module("pack_corpus")
    genes = ["9606:" + g for g in "ABCDEF"] + ["559292:YAL001C"]
    features = tmp_path / "features"
    features.mkdir()
    (features / "genes.json").write_text(json.dumps(genes))
    rng = np.random.default_rng(42)
    for name, value in [
        ("sequence.npy", rng.normal(size=(7, 12)).astype(np.float32)),
        ("annotation.npy", rng.normal(size=(7, 4)).astype(np.float32)),
        ("known.npy", np.ones((7, 2), bool)),
    ]:
        np.save(features / name, value, allow_pickle=False)
    (features / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "slp.static-features/v2",
                "files": {p.name: data.digest(p) for p in features.iterdir()},
                "fitted_human_intervention_genes": [],
            }
        )
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    monkeypatch.setenv("SLP_JOB", "synthetic-fixture")

    def put(path, *, data=None, prep=False):
        (corpus / Path(path).name).write_bytes(data)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(pack, "request", put)
    monkeypatch.setattr(
        pack,
        "save",
        lambda name, value: (corpus / (name + ".json")).write_text(pack.encode(value)),
    )
    writer = pack.Writer(genes, "fixture")
    template = dict(
        source="fitness-fixture",
        study="fixture",
        taxon=9606,
        context="cell",
        assay="fitness",
        kind="fitness",
        units="score",
        scope="population",
        license="synthetic",
        training_allowed=True,
        lineage={"synthetic": True},
    )
    for i, g in enumerate(genes):
        writer.add(
            template={**template, "taxon": 559292 if g.startswith("559292:") else 9606},
            genes=[g],
            value=float(i),
            native_row=i,
            unit=i,
        )
    writer.finish()
    manifest = corpus / "fixture-manifest.json"
    (tmp_path / "data.json").write_text(
        json.dumps(
            {
                "packed": [
                    {
                        "path": "corpus/fixture-manifest.json",
                        "sha256": data.digest(manifest),
                    }
                ]
            }
        )
    )
    b = tmp_path / "benchmark"
    b.mkdir()
    pairs = [("E", "F", 0), ("E", "F", 1), ("C", "D", 0), ("C", "D", 1), ("A", "B", 1)]
    rows = [
        {
            "row": i,
            "targets": ["9606:" + a, "9606:" + c],
            "label": y,
            "context": "pan-cancer",
        }
        for i, (a, c, y) in enumerate(pairs)
    ]
    payload = gzip.compress(b"\n".join(json.dumps(r).encode() for r in rows))
    (b / "train.jsonl.gz").write_bytes(payload)
    (b / "train.json").write_text(
        json.dumps(
            {
                "rows": len(rows),
                "shards": [
                    {
                        "name": "train.jsonl.gz",
                        "bytes": len(payload),
                        "sha256": data.digest(b / "train.jsonl.gz"),
                    }
                ],
            }
        )
    )
    fold = {
        "name": "fixture",
        "benchmark": "fixture",
        "protocol": {},
        "outer_held": genes[:2],
        "inner_held": genes[2:4],
        "partitions": {"train": "train.json", "test": "never-download-test.json"},
    }
    (b / "fold.json").write_text(json.dumps(fold))
    phase = dict(
        quantitative_weights={"fitness": 1.0},
        updates=2,
        batch_size=4,
        max_seconds=30,
        checkpoint_every=2,
        learning_rate=0.001,
        weight_decay=0.01,
        warmup=1,
        max_queries=3,
        retain_updates=[1, 2],
    )
    recipe = {
        "purpose": "disposable-readiness",
        "seed": 42,
        "cpu_threads": 1,
        "model": dict(
            width=32,
            layers=2,
            heads=4,
            sequence_dim=12,
            annotation_dim=4,
            context_dim=8,
        ),
        "pretrain": phase,
        "adapt": {**phase, "sl_fraction": 0.5},
    }
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))

    def run(stage, extra=()):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "modules/slp-1-2-r2/fit.py"),
                "--recipe",
                str(tmp_path / "recipe.json"),
                "--data",
                str(tmp_path / "data.json"),
                "--fold",
                str(b / "fold.json"),
                "--features",
                str(features),
                "--stage",
                stage,
                "--scope",
                "inner",
                "--deadline",
                str(time.time() + 900),
                "--device",
                "cpu",
                "--output",
                str(tmp_path / stage),
                *extra,
            ],
            text=True,
            capture_output=True,
        )

    result = run("pretrain")
    assert result.returncode == 0, result.stderr
    assert (
        torch.load(tmp_path / "pretrain/checkpoint-u000002.pt", weights_only=True)[
            "update"
        ]
        == 2
    )
    result = run("adapt", ["--initialize", str(tmp_path / "pretrain/checkpoint.pt")])
    assert result.returncode == 0, result.stderr
    exposure = json.loads((tmp_path / "adapt/exposure.json").read_text())
    assert exposure["views"][0]["fitting_rows"] == 2
    identity = json.loads((tmp_path / "adapt/identity.json").read_text())
    assert identity["human_fitted_intervention_genes"] == genes[4:6]
    score_command = [
        sys.executable,
        str(ROOT / "modules/slp-1-2-r2/score.py"),
        "--checkpoint",
        str(tmp_path / "adapt/checkpoint.pt"),
        "--features",
        str(features),
        "--fold",
        str(b / "fold.json"),
        "--basal",
        str(tmp_path / "adapt/basal.json"),
        "--output",
        str(tmp_path / "scores"),
        "--candidate",
        "synthetic",
        "--scope",
        "inner",
        "--device",
        "cpu",
    ]
    scored = subprocess.run(score_command, text=True, capture_output=True)
    assert scored.returncode == 0, scored.stderr
    assert json.loads((tmp_path / "scores/metrics.json").read_text())["n"] == 2
    (tmp_path / "adapt/basal.json").write_text('{"changed":true}')
    changed = subprocess.run(score_command, text=True, capture_output=True)
    assert changed.returncode != 0 and "basal observations changed" in changed.stderr
    checkpoint = torch.load(tmp_path / "pretrain/checkpoint.pt", weights_only=True)
    checkpoint["identity"]["human_fitted_intervention_genes"].append(genes[0])
    torch.save(checkpoint, tmp_path / "bad.pt")
    # Initialization may not inherit an outer-held exposure, even if current data mask is clean.
    result = run(
        "adapt",
        [
            "--initialize",
            str(tmp_path / "bad.pt"),
            "--output",
            str(tmp_path / "bad-adapt"),
        ],
    )
    assert result.returncode != 0 and "exposure contract" in result.stderr


def test_direct_baselines_are_symmetric_inductive_and_trainable():
    baselines = module("baselines")
    net, b = fixture()
    b["query_kind"].fill_(4)
    b["target"] = torch.zeros((2, 4))
    b["target"][1] = 1
    for kind in ("feature_mlp", "sequence_similarity"):
        baseline = baselines.PairBaseline(net.config, kind).eval()
        first = baseline(b)["sl_logit"]
        torch.testing.assert_close(
            first, baseline(permute(b, "action", [1, 0]))["sl_logit"]
        )
        train.objective(baseline(b), b).backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in baseline.parameters()
        )


def test_diagnostic_variability_does_not_replace_accuracy():
    diagnostics = module("diagnostics")
    good = diagnostics.numeric_metrics(
        [0, 1, 2], [0, 1, 2], fitting_mean=1, wrong_gene_prediction=[2, 1, 0]
    )
    bad = diagnostics.numeric_metrics(
        [0, 1, 2], [100, 0, -100], fitting_mean=1, wrong_gene_prediction=[-100, 0, 100]
    )
    assert (
        good["mse_skill_over_fitting_mean"] == 1 and good["wrong_minus_correct_mse"] > 0
    )
    assert bad["mse_skill_over_fitting_mean"] < 0


def test_all_fold_bitmask_matches_explicit_native_species_filter(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "data", data)
    monkeypatch.setitem(sys.modules, "cloud_io", module("cloud_io"))
    audit = module("audit_folds")
    pack = prep_module("pack_corpus")
    genes = ["9606:A", "9606:B", "559292:YAL001C"]
    rows = np.zeros(3, pack.DTYPE)
    rows["targets"] = -1
    rows["targets"][:, 0] = np.arange(3)
    rows["template"] = [0, 0, 1]
    np.save(tmp_path / "units.npy", rows)
    templates = [
        dict(taxon=t, kind="rna", training_allowed=True) for t in (9606, 559292)
    ]
    protocols = [
        dict(name="f", scope="inner", forbidden=genes[:2]),
        dict(name="f", scope="outer", forbidden=genes[:1]),
    ]
    result = audit.audit(
        protocols, genes, [(tmp_path / "units.npy", templates, "synthetic")]
    )
    assert [v["fitting_pretrain"] for v in result["sources"][0]["views"]] == [1, 2]
    assert [v["fitting_human_adaptation"] for v in result["sources"][0]["views"]] == [
        0,
        1,
    ]


def test_checkpoint_publication_captures_atomic_snapshot_without_credentials(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    net, b = fixture()
    optimizer = torch.optim.AdamW(net.parameters())
    sampler = data.ConditionSampler([row("x", ["C"])], {"fitness": 1.0})
    path = tmp_path / "checkpoint.pt"
    train.save_checkpoint(
        path, net, optimizer, sampler, update=2, identity={"purpose": "synthetic"}
    )
    expected = data.digest(path)

    def publish(directory, job):
        snapshot = Path(directory) / "checkpoint.pt"
        assert data.digest(snapshot) == expected
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"next atomic checkpoint")
        replacement.replace(path)
        assert data.digest(snapshot) == expected
        return {"files": {"checkpoint.pt": {"sha256": expected}}}

    monkeypatch.setitem(
        sys.modules,
        "cloud_io",
        SimpleNamespace(upload_directory=publish, download_directory=None),
    )
    monkeypatch.setitem(sys.modules, "data", data)
    monkeypatch.setitem(sys.modules, "train", train)
    spec = importlib.util.spec_from_file_location(
        "checkpoint_store_test", ROOT / "scripts/slp12_r2_checkpoint_store.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    before = list(sys.path)
    try:
        helper.main(
            SimpleNamespace(
                action="publish",
                path=str(path),
                job="synthetic",
                module_directory=str(ROOT / "modules/slp-1-2-r2"),
            )
        )
    finally:
        sys.path[:] = before
    assert path.read_bytes() == b"next atomic checkpoint"
