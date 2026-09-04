"""OMF protocol boundary for the typed sparse SLp-1.1 candidate.

Phase 1 validates and materializes the corpus/model contract. It deliberately
does not claim to train or select a biological checkpoint.
"""

from __future__ import annotations

from collections import Counter

from omf.sdk import ProtocolRequest, ProtocolResult, main


def _validation_outputs() -> dict[str, object]:
    return {
        "contractState": {
            "schema": "slp.world-sparse-contract/v1.1",
            "validationOnly": True,
        },
        "parameterCount": 0,
        "recordCount": 0,
        "targetValueCount": 0,
        "sourceScheduleCounts": {},
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    corpus_input = request.inputs.get("corpus")
    from slp_sparse_corpus import CorpusIndex, pinned_dataset_path

    corpus_path = pinned_dataset_path(corpus_input)
    from slp_sparse_architecture import SparseTypedWorldModel

    corpus = CorpusIndex.load(corpus_path)
    config = request.config
    model = SparseTypedWorldModel(
        corpus.world_config(
            d_model=int(config.get("dModel", 64)),
            nhead=int(config.get("nhead", 4)),
            encoder_layers=int(config.get("encoderLayers", 2)),
            decoder_layers=int(config.get("decoderLayers", 1)),
            ffn_multiplier=int(config.get("ffnMultiplier", 4)),
            dropout=float(config.get("dropout", 0.0)),
        )
    )
    record_count = sum(item.records for item in corpus.shards)
    target_count = sum(item.target_values for item in corpus.shards)
    draws = int(config.get("scheduleRecords", record_count))
    schedule = corpus.record_sampler(int(config.get("seed", 731))).schedule(draws)
    schedule_counts = Counter()
    shard_cache = {}
    for location in schedule:
        if location.shard_index not in shard_cache:
            shard_cache[location.shard_index] = corpus.load_shard(location.shard_index)
        shard = shard_cache[location.shard_index]
        source = int(shard.arrays["source_index"][location.row_index])
        schedule_counts[corpus.sources[source]] += 1
    state = {
        "schema": "slp.world-sparse-contract/v1.1",
        "corpusSchema": "slp.corpus/v1.1",
        "modelFormat": "slp.world-sparse/v1.1",
        "datasetId": corpus.dataset_id,
        "datasetVersion": corpus.version,
        "modelConfig": model.config.as_dict(),
        "dictionarySizes": {
            "entities": len(corpus.entity_id),
            "queries": len(corpus.query_id),
            "panels": len(corpus.panel_id),
        },
        "trainingImplemented": False,
    }
    outputs = {
        "contractState": state,
        "parameterCount": model.count_parameters(),
        "recordCount": record_count,
        "targetValueCount": target_count,
        "sourceScheduleCounts": dict(sorted(schedule_counts.items())),
    }
    return ProtocolResult(status="ok", outputs=outputs, state=state)


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
