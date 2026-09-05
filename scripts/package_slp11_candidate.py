"""Create an immutable local development package with a tested inference runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",required=True)
    parser.add_argument("--features",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    candidate, output = Path(args.candidate), Path(args.output)
    protocol = json.loads((candidate/"protocol.json").read_text(encoding="utf-8"))
    expected_features = protocol.get("featuresSha256", protocol.get("features_sha256"))
    if not expected_features or sha(args.features) != expected_features:
        raise ValueError("static features must exactly match the frozen candidate protocol")
    ensemble = (candidate/"ensemble-manifest.json").exists()
    output.mkdir(parents=True,exist_ok=False)
    source = Path(__file__).resolve().parents[1]/"modules/slp-1-1-world-transition-v1"
    (output/"runtime").mkdir()
    for name in ("inference.py","transition_model.py","ensemble_inference.py","predict.py"):
        frozen = candidate/"source"/name
        runtime_source = frozen if ensemble and name != "predict.py" else source/name
        shutil.copyfile(runtime_source,output/"runtime"/name)
    shutil.copyfile(source/"requirements.lock",output/"requirements.lock")
    if ensemble:
        shutil.copytree(candidate/"members", output/"members")
        for name in ("ensemble-manifest.json", "ensemble-exposure-uncertainty.npz"):
            shutil.copyfile(candidate/name,output/name)
    else:
        for name in ("model.safetensors","model-config.json","reference.npz"):
            shutil.copyfile(candidate/name,output/name)
        if (candidate/"linear-reference.npz").exists():
            shutil.copyfile(candidate/"linear-reference.npz",output/"linear-reference.npz")
        exposure = candidate/"world-exposure-uncertainty.npz"
        if not exposure.exists():
            exposure = candidate/"exposure-uncertainty.npz"
        shutil.copyfile(exposure,output/"exposure-uncertainty.npz")
    for name in ("protocol.json", "report.json"):
        shutil.copyfile(candidate/name,output/name)
    shutil.copyfile(args.features,output/"static-features.npz")
    (output/"README.md").write_text(
        "# SLp-1.1 local molecular development candidate\n\n"
        "Experimental human single-gene CRISPRi predictions for the recorded K562/RPE1 contexts.\n"
        "This package is not a certified release or an SL predictor. Scientific results are recorded in report.json.\n\n"
        "Use `python runtime/predict.py --gene ENSG00000012048 --context replogle-2022-k562-essential-day-6`.\n"
        "Query outputs are aggregate core-control-standardized molecular measurements. `--num-cells` changes measurement uncertainty only.\n"
        "The numerical inference API supports externally encoded static features; this convenience CLI uses the packaged cache.\n",
        encoding="utf-8")
    manifest = {"status":"local-experimental-candidate-not-certified-release",
                "parent_candidate":str(candidate.resolve()),
                "parent_report_sha256":sha(candidate/"report.json"),
                "files":{str(p.relative_to(output)).replace("\\","/"):sha(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    (output/"package-manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
