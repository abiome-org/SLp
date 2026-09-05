#!/usr/bin/env python3
"""Run a capacity-matched source3 neural BP action-feature experiment pair."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/slp11-transition/human-source3-vs-four-context-mean-objective-seed731-v2"
DATA = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
PHYSICAL = ROOT / "data/derived/slp11-human-physical/direct-experiments700-v1/human-esm-go-physical-features.npz"
BP = ROOT / "data/derived/slp11-human-go-bp/goa-2022-09-19-ensembl108-source3-fit-svd128-v1/human-go-bp-source3-fit-svd128-features.npz"
FROZEN_REFERENCE = OLD / "frozen-reference.npz"
MODEL = OLD / "source/control_transition_model.py"
MEAN_OBJECTIVE = OLD / "source/mean_objective.py"
WEIGHTING = OLD / "source/objective_weighting.py"
METRICS = OLD / "source/four_context_baselines.py"
INFERENCE = OLD / "source/four_context_mean_inference.py"
VERIFIER = OLD / "source/verify_artifact.py"
BP_RIDGE = ROOT / "results/slp11-transition/human-gwps-bp-ridge-source3-seed731-v2/report.json"
BP_KERNEL = ROOT / "results/slp11-transition/human-gwps-bp-nystrom-rbf512-seed731-v1/report.json"
OLD_MEAN = OLD / "arm-source3/report.json"
OUTPUT = ROOT / "results/slp11-transition/human-source3-bp-neural-mean-pair-seed731-v2"
PROFILE = ROOT / "results/slp11-transition/human-source3-bp-neural-mean-profile-v2.json"
HASHES = {
    "data": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "physical": "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    "bp": "b29cbd70f08e227cddfc013e66cd1032212c8cb62e6e25162965a57101cd1fac",
    "reference": "54cac4bc2e2ee02a6d78f812d5646cf3988154d5ae4f371265b24751f03c99b1",
    "model": "fdb4555bd0f7c0a0786539da67048f6985f4ec2f36ef7aa45bd22c7c6bfbb2ef",
    "meanObjective": "bed662c89a8ec9b5dd47019fd22d09caf9cc6f56871e4b896fe0a43fb52061fa",
    "weighting": "2f54e3a3e6ef4e84b4d7ca63d62fd38bd0751a1f7e8aaf4769f9a2c505352c38",
    "metrics": "84d1d2d7c727d7612f0fbc1d9aa69659a20b864896184336071d4189c33b458c",
    "inference": "b2afc85136c7c53df23b48f5401b3ff282da2fcc6e97dc1aeeadbbbb3d008d0f",
    "verifier": "cbdde65c407cffefa5f99f88638936a2ea8e79bf6f0a0bea17bf78725528bcc5",
    "bpRidge": "8a3d1ba2265dc09bf6856c97c7a791775ef3282594beed269f708f353d895a0a",
    "bpKernel": "d8259c864460a21f9a13718b2190aad926ca58dc01409c0fab1220a6fbbd276c",
    "oldMean": "ac5e05603365f9eb2c8888cc8344a99d68c5d209bbfc7fac6df0d073ea75508f",
}
CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
ARMS = ("masked-bp-control", "bp128-present")
SEED = 731
BATCH = 64
STEPS = 12_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict): return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)): return [clean(entry) for entry in item]
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, float) and not np.isfinite(item): return None
        return item
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_inputs() -> dict[str, Path]:
    paths = {"data": DATA, "physical": PHYSICAL, "bp": BP, "reference": FROZEN_REFERENCE, "model": MODEL, "meanObjective": MEAN_OBJECTIVE, "weighting": WEIGHTING, "metrics": METRICS, "inference": INFERENCE, "verifier": VERIFIER, "bpRidge": BP_RIDGE, "bpKernel": BP_KERNEL, "oldMean": OLD_MEAN}
    for name, path in paths.items():
        if sha256(path) != HASHES[name]: raise ValueError(f"input hash mismatch: {name}")
    return paths


def initialize_extended(model_module, seed: int = SEED):
    torch.manual_seed(seed)
    old = model_module.MinimalControlTransition(model_module.Config(1156, 1188, hidden_dim=128, state_dim=128, dropout=.2))
    old_state = old.state_dict(); torch.manual_seed(seed)
    new = model_module.MinimalControlTransition(model_module.Config(1285, 1188, hidden_dim=128, state_dim=128, dropout=.2))
    with torch.no_grad():
        for name, target in new.state_dict().items():
            source = old_state[name]
            if target.shape == source.shape: target.copy_(source)
            elif name == "action_encoder.0.weight" and target.shape == (128, 1285):
                target.zero_(); target[:, :1156].copy_(source)
            else: raise ValueError(f"unexpected expanded parameter: {name}")
    return new, old


def load_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    verify_inputs()
    with np.load(DATA, allow_pickle=False) as archive: data = {name: archive[name] for name in archive.files}
    with np.load(FROZEN_REFERENCE, allow_pickle=False) as archive: frozen = {name: archive[name] for name in archive.files}
    with np.load(PHYSICAL, allow_pickle=False) as archive:
        physical_lookup = dict(zip(archive["entity_id"].astype(str), archive["feature_values"], strict=True))
    with np.load(BP, allow_pickle=False) as archive:
        if not np.all(archive["entity_taxon"] == 9606): raise ValueError("BP taxonomy drift")
        bp_lookup = {str(gene): (value, present) for gene, value, present in zip(archive["entity_id"], archive["feature_values"], archive["annotation_present"], strict=True)}
    if len(data["split_test"]) or tuple(data["context_ids"].astype(str)) != CONTEXTS: raise ValueError("source3 split/context drift")
    if not np.array_equal(data["query_ids"], frozen["query_ids"]): raise ValueError("query order drift")
    action_ids = data["action_ids"].astype(str)
    physical = np.stack([physical_lookup[gene] for gene in action_ids]).astype(np.float32)
    bp = np.stack([bp_lookup[gene][0] for gene in action_ids]).astype(np.float32)
    presence = np.asarray([bp_lookup[gene][1] for gene in action_ids], dtype=np.float32)[:, None]
    fitting_genes = np.unique(action_ids[data["split_train"]])
    if len(fitting_genes) != 6866: raise ValueError("source3 fitting gene roster drift")
    fitting_bp = np.stack([bp_lookup[gene][0] for gene in fitting_genes]).astype(np.float64)
    bp_mean = fitting_bp.mean(0).astype(np.float32); bp_std = fitting_bp.std(0).astype(np.float32)
    bp_std = np.where(bp_std > 1e-5, bp_std, 1.0).astype(np.float32)
    actions = {
        "masked-bp-control": np.concatenate([physical, np.zeros((len(physical), 129), np.float32)], axis=1),
        "bp128-present": np.concatenate([physical, bp, presence], axis=1),
    }
    references = {}
    for arm in ARMS:
        tail_mean = np.zeros(129, np.float32) if arm == "masked-bp-control" else np.concatenate([bp_mean, np.zeros(1, np.float32)])
        tail_std = np.ones(129, np.float32) if arm == "masked-bp-control" else np.concatenate([bp_std, np.ones(1, np.float32)])
        reference = dict(frozen)
        reference["feature_mean"] = np.concatenate([frozen["feature_mean"], tail_mean]).astype(np.float32)
        reference["feature_std"] = np.concatenate([frozen["feature_std"], tail_std]).astype(np.float32)
        for name in ("control_mean", "context_values", "context_mask", "context_ids", "objective_query_scale"):
            reference[name] = frozen[name][:3]
        references[arm] = reference
    audit = {"fittingGenes": len(fitting_genes), "bpMeanSha256": hashlib.sha256(bp_mean.tobytes()).hexdigest(), "bpStdSha256": hashlib.sha256(bp_std.tobytes()).hexdigest(), "fittingGeneRosterSha256": hashlib.sha256("\n".join(fitting_genes).encode()).hexdigest(), "annotatedFittingGenes": int(sum(bool(bp_lookup[gene][1]) for gene in fitting_genes)), "baseNormalizerBitExact": all(np.array_equal(references[arm]["feature_mean"][:1156], frozen["feature_mean"]) and np.array_equal(references[arm]["feature_std"][:1156], frozen["feature_std"]) for arm in ARMS)}
    return data, actions, references, audit


def forward(model, raw_actions, contexts, reference, device):
    normalized = (raw_actions - reference["feature_mean"]) / reference["feature_std"]
    query = (reference["query_features"] - reference["query_feature_mean"]) / reference["query_feature_std"]
    selected = reference["context_query_indices"]
    return model(torch.as_tensor(normalized, device=device), torch.as_tensor(query, device=device), torch.as_tensor(reference["control_mean"][contexts], device=device), torch.as_tensor(reference["delta_amplitude"], device=device), torch.as_tensor(reference["objective_query_scale"][contexts], device=device), torch.as_tensor(query[selected], device=device), torch.as_tensor(reference["context_values"][contexts], device=device), torch.as_tensor(reference["context_mask"][contexts], dtype=torch.bool, device=device))


def make_runtime(raw_actions, reference, device):
    normalized = ((raw_actions - reference["feature_mean"]) / reference["feature_std"]).astype(np.float32)
    query = ((reference["query_features"] - reference["query_feature_mean"]) / reference["query_feature_std"]).astype(np.float32)
    selected = reference["context_query_indices"]
    return {
        "actions": torch.as_tensor(normalized, device=device),
        "query": torch.as_tensor(query, device=device),
        "control": torch.as_tensor(reference["control_mean"], device=device),
        "amplitude": torch.as_tensor(reference["delta_amplitude"], device=device),
        "scale": torch.as_tensor(reference["objective_query_scale"], device=device),
        "basal_features": torch.as_tensor(query[selected], device=device),
        "basal_values": torch.as_tensor(reference["context_values"], device=device),
        "basal_mask": torch.as_tensor(reference["context_mask"], dtype=torch.bool, device=device),
    }


def forward_runtime(model, rows, contexts, runtime, device):
    row_tensor = torch.as_tensor(rows, dtype=torch.int64, device=device)
    context_tensor = torch.as_tensor(contexts, dtype=torch.int64, device=device)
    return model(
        runtime["actions"][row_tensor], runtime["query"],
        runtime["control"][context_tensor], runtime["amplitude"],
        runtime["scale"][context_tensor], runtime["basal_features"],
        runtime["basal_values"][context_tensor], runtime["basal_mask"][context_tensor],
    )


def initial_parity(model_module, data, actions, references, device):
    train = data["split_train"][:2]; differences = {}
    for arm in ARMS:
        new, old = initialize_extended(model_module); new = new.to(device).eval(); old = old.to(device).eval()
        old_raw = actions[arm][train, :1156]; old_ref = dict(references[arm]); old_ref["feature_mean"] = old_ref["feature_mean"][:1156]; old_ref["feature_std"] = old_ref["feature_std"][:1156]
        with torch.no_grad():
            old_output = forward(old, old_raw, data["context_index"][train], old_ref, device)
            new_output = forward(new, actions[arm][train], data["context_index"][train], references[arm], device)
        maximum = max(float(torch.max(torch.abs(old_output[name] - new_output[name])).cpu()) for name in ("mean", "delta", "state", "basal_state", "intervention_delta"))
        if maximum > 1e-5: raise ValueError(f"initial parity failed: {arm} {maximum}")
        differences[arm] = maximum
    return differences


def profile(device_name: str, steps: int) -> dict[str, object]:
    _, _, references, _ = load_data(); model_module = load(MODEL, "bp_profile_model"); device = torch.device(device_name)
    model, _ = initialize_extended(model_module); model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.1); reference = references[ARMS[0]]
    raw = np.zeros((BATCH, 1285), np.float32); contexts = np.arange(BATCH) % 3; target = torch.zeros((BATCH,7036),device=device); observed = torch.ones_like(target,dtype=torch.bool); runtime = make_runtime(raw, reference, device); rows = np.arange(BATCH)
    objective = load(MEAN_OBJECTIVE, "bp_profile_objective"); torch.use_deterministic_algorithms(True)
    torch.cuda.synchronize() if device.type == "cuda" else None; started = time.monotonic()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True); result = forward_runtime(model, rows, contexts, runtime, device)
        loss = objective.masked_standardized_mse(result["mean"], target, observed, torch.ones_like(target), torch.ones(BATCH,device=device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
    torch.cuda.synchronize() if device.type == "cuda" else None; elapsed = time.monotonic()-started
    return {"schema":"slp.source3-bp-neural-profile/v1","device":device_name,"targetFreeSynthetic":True,"steps":steps,"seconds":elapsed,"secondsPerStep":elapsed/steps,"projectedSecondsPer12000StepArm":elapsed/steps*STEPS,"bothArmsProjectedSeconds":elapsed/steps*STEPS*2,"batch":BATCH,"queries":7036,"actionFeatures":1285}


def prepare(output: Path, profile_path: Path):
    if output.exists(): raise FileExistsError(output)
    paths = verify_inputs(); data, actions, references, audit = load_data(); model_module = load(MODEL,"bp_prepare_model"); parity = initial_parity(model_module,data,actions,references,torch.device("cpu")); profile_report=json.loads(profile_path.read_text())
    if profile_report["bothArmsProjectedSeconds"] >= 540: raise ValueError("pair profile exceeds compute allowance")
    output.mkdir(parents=True); source=output/"source"; source.mkdir()
    sources={"control_transition_model.py":MODEL,"mean_objective.py":MEAN_OBJECTIVE,"objective_weighting.py":WEIGHTING,"four_context_baselines.py":METRICS,"four_context_mean_inference.py":INFERENCE,"verify_artifact.py":VERIFIER,"trainer.py":Path(__file__)}; source_hashes={}
    for name,path in sources.items(): shutil.copy2(path,source/name); source_hashes[name]=sha256(source/name)
    shutil.copy2(FROZEN_REFERENCE,output/"frozen-reference-source.npz")
    np.savez_compressed(output/"bp-normalizer.npz", mean=references["bp128-present"]["feature_mean"][1156:1284], std=references["bp128-present"]["feature_std"][1156:1284])
    protocol={"schema":"slp.source3-bp-neural-mean-pair-protocol/v2","hypothesis":"BP action annotations improve a fixed source3 neural mean model beyond a capacity-matched masked-BP control and BP linear ridge in every context.","arms":{"masked-bp-control":"physical1156 plus129 all-zero inputs","bp128-present":"physical1156 plus fitting-gene-normalized BP128 plus raw availability bit"},"isolation":"both models use action width1285, identical old-compatible seed731 tensors, and zero new129 input weights; data/query/basal/objective/optimizer fixed","bpNormalization":"mean/population SD on union6866 source3 fitting action genes only; SD<=1e-5 replaced1; presence bit raw and not normalized","training":"12000 updates,B64,seed731 deterministic shuffled fitting rows,global equal-context/equal-gene weights,all7036queries,masked standardized mean MSE,AdamWlr.0005/decay.1,clip1,final checkpoint only,no validation before both arms final","gate":"BP arm each3 contexts: >=2% raw gene-profile MSE improvement versus matched masked-BP control and BP linear ridge; centered r finite>=.10 and nonregression versus both","descriptiveFrontier":"BP kernel, original neural v2, and prior fixed-step source3 mean model; no winner claim if stronger baseline wins","profile":profile_report,"inputAudit":audit,"initialOutputParityMaxDifference":parity,"inputs":{name:{"path":str(path),"sha256":HASHES[name]} for name,path in paths.items()},"sourceHashes":source_hashes,"supersedesPreparedOnly":{"path":"results/slp11-transition/human-source3-bp-neural-mean-pair-seed731-v1","reason":"added explicit deterministic-algorithm enforcement and executing-source checksum before any biological fitting"},"hepg2OutcomesUsed":False,"jurkatAccessed":False,"testAccessed":False,"benchmarkAccessed":False}
    write_json(output/"protocol.json",protocol)
    for arm in ARMS:
        arm_path=output/arm; arm_path.mkdir(); write_json(arm_path/"protocol.json",{"schema":"slp.source3-bp-neural-arm-protocol/v1","arm":arm,"pairProtocolSha256":sha256(output/"protocol.json"),"validationEvaluations":1,"validationTiming":"only after both final checkpoints"})
    prepared={"protocolSha256":sha256(output/"protocol.json"),"armProtocolSha256":{arm:sha256(output/arm/"protocol.json") for arm in ARMS},"sourceHashes":source_hashes}; write_json(output/"PREPARED.json",prepared); return prepared


def fit_arm(arm,data,raw_actions,reference,output,device,model_module,objective,weighting):
    arm_path=output/arm; train=data["split_train"]; contexts=data["context_index"][train]; weights=weighting.training_row_weights(contexts,data["action_ids"][train],objective=weighting.EQUAL_CONTEXT_GENE_V1).astype(np.float32)
    model,_=initialize_extended(model_module); model=model.to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.1)
    target=torch.as_tensor(data["targets"][train],device=device); observed=torch.as_tensor(data["observed"][train],device=device); scale=torch.as_tensor(reference["objective_query_scale"],device=device); weight=torch.as_tensor(weights,device=device); actions=raw_actions[train]; runtime=make_runtime(actions,reference,device)
    batches=objective.deterministic_shuffled_batches(np.arange(len(train)),batch_size=BATCH,steps=STEPS,seed=SEED); losses=[]; started=time.monotonic(); model.train()
    for step,rows in enumerate(batches,1):
        optimizer.zero_grad(set_to_none=True); local=contexts[rows]; result=forward_runtime(model,rows,local,runtime,device); loss=objective.masked_standardized_mse(result["mean"],target[rows],observed[rows],scale[local],weight[rows]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); losses.append(float(loss.detach()))
        if step%1000==0: print(json.dumps({"arm":arm,"step":step,"recentLoss":float(np.mean(losses[-100:])),"seconds":time.monotonic()-started}),flush=True)
        if time.monotonic()-started>270: raise TimeoutError(f"arm exceeded270sec: {arm}")
    save_file({name:value.detach().cpu() for name,value in model.state_dict().items()},str(arm_path/"model.safetensors")); np.savez_compressed(arm_path/"reference.npz",**reference)
    return model,{"arm":arm,"updates":STEPS,"seconds":time.monotonic()-started,"finalRecentLoss":float(np.mean(losses[-100:]))}


def predict(model, raw, context, reference, device):
    model.eval(); chunks=[]; runtime=make_runtime(raw,reference,device)
    with torch.no_grad():
        for start in range(0,len(raw),256):
            rows=np.arange(start,min(start+256,len(raw))); chunks.append(forward_runtime(model,rows,context[rows],runtime,device)["mean"].cpu().numpy())
    return np.concatenate(chunks)


def package_arm(arm,model,data,raw,reference,output,device):
    arm_path=output/arm; train=data["split_train"]; probe=train[:6]; expected=predict(model,raw[probe],data["context_index"][probe],reference,device)
    np.savez_compressed(arm_path/"target-free-probe.npz",raw_action_features=raw[probe],context_index=data["context_index"][probe],expected_mean=expected)
    source_hashes = {
        f"../source/{path.name}": sha256(path)
        for path in sorted((output / "source").glob("*.py"))
    }
    manifest={"schema":"slp.source3-bp-neural-arm-artifact/v1","sha256":{"model.safetensors":sha256(arm_path/"model.safetensors"),"reference.npz":sha256(arm_path/"reference.npz"),"target-free-probe.npz":sha256(arm_path/"target-free-probe.npz"),**source_hashes}}
    write_json(arm_path/"artifact-manifest.json",manifest); verification=subprocess.run([sys.executable,str(output/"source/verify_artifact.py"),str(arm_path)],capture_output=True,text=True,check=True,cwd=ROOT)
    return json.loads(verification.stdout)


def score_arm(arm,model,data,raw,reference,output,device,metrics):
    validation=data["split_validation"]; row_prediction=predict(model,raw[validation],data["context_index"][validation],reference,device); result={}; arrays={}
    for index,name in enumerate(CONTEXTS):
        local=data["context_index"][validation]==index; genes,prediction,mask,_=metrics.collapse_equal_records(data["action_ids"][validation][local],row_prediction[local],data["observed"][validation][local]); tg,truth,tm,_=metrics.collapse_equal_records(data["action_ids"][validation][local],data["targets"][validation][local],data["observed"][validation][local])
        if not np.array_equal(genes,tg) or not np.array_equal(mask,tm): raise ValueError("validation collapse drift")
        value=metrics.point_metrics(prediction,truth,mask,reference["objective_query_scale"][index],data["targets"][data["split_train"]][data["context_index"][data["split_train"]]==index].mean(0)); result[name]=value
        arrays[f"context{index}_action_ids"]=genes; arrays[f"context{index}_prediction"]=prediction.astype(np.float32); arrays[f"context{index}_truth"]=truth.astype(np.float32); arrays[f"context{index}_observed"]=mask
    np.savez_compressed(output/arm/"predictions.npz",**arrays); return result


def decide(reports):
    ridge=json.loads(BP_RIDGE.read_text()); kernel=json.loads(BP_KERNEL.read_text()); old_mean=json.loads(OLD_MEAN.read_text()); contexts={}; all_pass=True
    for name in CONTEXTS:
        control=reports[ARMS[0]]["metrics"][name]; bp=reports[ARMS[1]]["metrics"][name]; ridge_score=ridge["contexts"][name]["arms"]["physical1156_bp128_present1"]["scores"]; kernel_context=kernel["contexts"][name]
        bp_mse=bp["gene_profile_raw_mse"]; bp_r=bp["independently_query_centered_profile_pearson"]; control_r=control["independently_query_centered_profile_pearson"]; ridge_r=ridge_score["independentlyQueryCenteredPearson"]
        finite = lambda value: value is not None and np.isfinite(float(value))
        checks={"mseVsMaskedControlAtLeast002":1-bp_mse/control["gene_profile_raw_mse"]>=.02,"mseVsBpRidgeAtLeast002":1-bp_mse/ridge_score["geneProfileMse"]>=.02,"allRequiredRFinite":all(finite(x) for x in (bp_r,control_r,ridge_r)),"rAtLeast010":finite(bp_r) and bp_r>=.1,"rNonregressionVsMaskedControl":finite(bp_r) and finite(control_r) and bp_r>=control_r,"rNonregressionVsBpRidge":finite(bp_r) and finite(ridge_r) and bp_r>=ridge_r}; passed=all(checks.values()); all_pass &= passed
        contexts[name]={"checks":checks,"passed":passed,"maskedControl":control,"bpNeural":bp,"bpLinearRidge":ridge_score,"bpKernel":kernel_context["candidate"],"oldNeuralV2":kernel_context["comparators"]["minimalControlV2"],"oldFixedStepMean":old_mean["validationMetrics"][name],"bpKernelBeatsNeuralMse":kernel_context["candidate"]["geneProfileMse"]<bp_mse}
    return {"contexts":contexts,"passed":bool(all_pass)}


def execute(output:Path,device_name:str):
    prepared=json.loads((output/"PREPARED.json").read_text());
    if sha256(output/"protocol.json")!=prepared["protocolSha256"]: raise ValueError("protocol drift")
    for name,digest in prepared["sourceHashes"].items():
        if sha256(output/"source"/name)!=digest: raise ValueError(f"source drift:{name}")
    if sha256(Path(__file__)) != prepared["sourceHashes"]["trainer.py"]: raise ValueError("executing trainer differs from frozen source")
    torch.use_deterministic_algorithms(True)
    data,actions,references,audit=load_data(); device=torch.device(device_name); model_module=load(output/"source/control_transition_model.py","bp_train_model"); objective=load(output/"source/mean_objective.py","bp_train_objective"); weighting=load(output/"source/objective_weighting.py","bp_train_weighting"); metrics=load(output/"source/four_context_baselines.py","bp_metrics")
    parity=initial_parity(model_module,data,actions,references,device); models={}; reports={}
    for arm in ARMS: models[arm],reports[arm]=fit_arm(arm,data,actions[arm],references[arm],output,device,model_module,objective,weighting)
    write_json(output/"FROZEN-BEFORE-VALIDATION.json",{"protocolSha256":prepared["protocolSha256"],"models":{arm:sha256(output/arm/"model.safetensors") for arm in ARMS},"initialParity":parity,"validationEvaluations":0})
    for arm in ARMS:
        reports[arm]["portableReload"]=package_arm(arm,models[arm],data,actions[arm],references[arm],output,device); reports[arm]["metrics"]=score_arm(arm,models[arm],data,actions[arm],references[arm],output,device,metrics); reports[arm]["artifacts"]={name:sha256(output/arm/name) for name in ("model.safetensors","reference.npz","predictions.npz","artifact-manifest.json")}; write_json(output/arm/"report.json",reports[arm])
    decision=decide(reports); result={"schema":"slp.source3-bp-neural-mean-pair-result/v1","arms":reports,"decision":decision,"inputAudit":audit,"initialOutputParityMaxDifference":parity,"protocolSha256":prepared["protocolSha256"],"hepg2OutcomesUsed":False,"jurkatAccessed":False,"testAccessed":False,"benchmarkAccessed":False}; write_json(output/"report.json",result); return result


def main():
    parser=argparse.ArgumentParser(); modes=parser.add_mutually_exclusive_group(required=True); modes.add_argument("--profile",action="store_true"); modes.add_argument("--prepare-only",action="store_true"); modes.add_argument("--run",action="store_true"); parser.add_argument("--output",type=Path,default=OUTPUT); parser.add_argument("--profile-path",type=Path,default=PROFILE); parser.add_argument("--profile-steps",type=int,default=30); parser.add_argument("--device",choices=("cpu","cuda"),default="cuda"); args=parser.parse_args()
    if args.profile: result=profile(args.device,args.profile_steps); write_json(args.profile_path,result)
    elif args.prepare_only: result=prepare(args.output,args.profile_path)
    else: result=execute(args.output,args.device)
    print(json.dumps(result,sort_keys=True,default=lambda x:x.item()))


if __name__=="__main__": main()
