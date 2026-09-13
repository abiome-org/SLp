"""SL pair inference from a complete exported bundle; no label file is needed."""

import argparse
import json
from pathlib import Path
import torch
from artifact import load
from data import BatchBuilder, Scales
from records import Record


def predict(directory, pairs, *, context="pan-cancer", device="cpu", batch_size=64):
    model, features, vocabulary, basal, receipt = load(directory, device)
    rows = []
    for i, pair in enumerate(pairs):
        if len(pair) != 2 or len(set(pair)) != 2:
            raise ValueError("Each query needs two distinct canonical human genes")
        rows.append(
            Record(
                record_id=str(i),
                source_id="inference",
                study_id="inference",
                experiment_id=str(i),
                replicate_id="query",
                taxon=9606,
                targets=tuple(sorted(pair)),
                context=context,
                assay="human-SL",
                kind="sl",
                units="binary-SL-call",
                value=0.0,
                query_gene=None,
                mechanism="unknown",
                method="unknown",
                label_scope="pan-cancer" if context == "pan-cancer" else "cell-line",
                role="sl_label",
                license="user-query",
                raw_object="inference-query",
                raw_locator=str(i),
            )
        )
    builder = BatchBuilder(features, Scales([]), vocabulary, model.config, basal)
    values = []
    with torch.inference_mode():
        for lo in range(0, len(rows), batch_size):
            batch = {
                k: v.to(device)
                for k, v in builder([[r] for r in rows[lo : lo + batch_size]]).items()
            }
            values.extend(
                model(batch)["sl_logit"][:, 0].sigmoid().float().cpu().tolist()
            )
    return [
        {"targets": list(pair), "sl_probability": p} for pair, p in zip(pairs, values)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", default="pan-cancer")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = predict(
        args.bundle,
        json.loads(Path(args.pairs).read_text()),
        context=args.context,
        device=args.device,
    )
    Path(args.output).write_text(json.dumps(result, allow_nan=False))
