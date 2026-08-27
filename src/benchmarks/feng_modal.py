"""Feng 2024 SL models on Modal. Official splits, random negatives, 1:1."""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

ZENODO = "https://zenodo.org/api/records/14025191/files"
SMALL_MD5 = "5a36306fac1e31352c6d4645c8a57a6a"
LARGE_MD5 = "4a12e566dddba09d72f05753c69f5e64"
SMALL_PARTS = ["aa", "ab", "ac", "ad", "ae"]
LARGE_PARTS = ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj"]
SPLITS = ["CV1", "CV2", "CV3"]
CPU_MODELS = ["GRSMF", "SL2MF"]
GPU_MODELS = ["CMFW", "DDGCN", "KG4SL", "SLMGAE", "NSF4SL", "GCATSL", "SLGNN", "PTGNN", "MGE4SL"]
VOL = "/vol"

app = modal.App("slp-feng")
vol = modal.Volume.from_name("slp-feng", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "tar")
    .pip_install("numpy==1.26.4")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "tensorflow[and-cuda]",
        "wandb",
        "scikit-learn",
        "scipy",
        "pandas",
        "tqdm",
        "networkx",
        "lmdb",
        "h5py",
        "torch-geometric",
        "tf-keras",
    )
    .pip_install("dgl", find_links="https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html")
    .pip_install("numpy==1.26.4")
    .env(
        {
            "WANDB_MODE": "offline",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "PYTHONUNBUFFERED": "1",
            "DGLBACKEND": "pytorch",
            "TF_USE_LEGACY_KERAS": "1",
        }
    )
)

if modal.is_local():
    _src = Path(__file__).resolve().parents[2] / "data" / "models" / "SL_benchmark" / "src"
    image = image.add_local_dir(
        str(_src),
        remote_path="/bench/src",
        copy=True,
        ignore=["**/wandb/**", "**/__pycache__/**", "**/*.pyc"],
    )


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _curl(name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["curl", "-fsSL", "--retry", "8", "--retry-all-errors", "-o", str(dest), f"{ZENODO}/{name}/content"]
    )


def _link_bench() -> None:
    Path("/bench").mkdir(exist_ok=True)
    for name in ("data", "results"):
        link = Path("/bench") / name
        target = Path(VOL) / name
        target.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)


def _extract(bundle: str) -> None:
    parts = SMALL_PARTS if bundle == "small" else LARGE_PARTS
    expect = SMALL_MD5 if bundle == "small" else LARGE_MD5
    marker = Path(VOL) / "data" / "preprocessed_data" / "human_sl_9845.csv"
    if bundle == "small" and marker.exists():
        return
    if bundle == "large" and (Path(VOL) / "data" / "preprocessed_data" / "pilsl_data").exists() and any(
        (Path(VOL) / "data" / "preprocessed_data" / "pilsl_data").iterdir()
    ):
        return
    scratch = Path(VOL) / "scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    files = []
    for p in parts:
        name = f"data_{bundle}.tar.gz.part{p}"
        dest = scratch / name
        _curl(name, dest)
        files.append(dest)
    tar = scratch / f"data_{bundle}.tar.gz"
    with tar.open("wb") as out:
        for f in files:
            with f.open("rb") as inp:
                shutil.copyfileobj(inp, out)
            f.unlink()
    got = _md5(tar)
    if got != expect:
        raise RuntimeError(f"{tar.name} md5 {got} != {expect}")
    unpack = scratch / "unpack"
    unpack.mkdir()
    subprocess.check_call(["tar", "-xzf", str(tar), "-C", str(unpack)])
    tar.unlink()
    inner = unpack / "data" if (unpack / "data" / "preprocessed_data").exists() else unpack
    dest = Path(VOL) / "data"
    dest.mkdir(parents=True, exist_ok=True)
    for item in inner.iterdir():
        target = dest / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))
    shutil.rmtree(scratch)


def _run(model: str, split: str) -> str:
    _link_bench()
    logs = Path(VOL) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{model}_{split}_Random_1.log"
    with log.open("w") as f:
        proc = subprocess.run(
            [sys.executable, "main.py", "-m", model, "-ns", "Random", "-ds", split, "-pn", "1"],
            cwd="/bench/src",
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    ok = Path(VOL) / "logs" / f"{model}_{split}_Random_1.ok"
    if proc.returncode == 0:
        ok.write_text("ok\n")
        status = f"ok {model} {split}"
    else:
        status = f"fail {model} {split} exit={proc.returncode}"
        fail_path = Path(VOL) / "logs" / "failures.txt"
        with fail_path.open("a") as ff:
            ff.write(status + "\n")
    vol.commit()
    return status


@app.function(image=image, timeout=4 * 3600, memory=65536, cpu=4.0, volumes={VOL: vol})
def hydrate(bundle: str = "small") -> str:
    _extract(bundle)
    vol.commit()
    data = Path(VOL) / "data"
    n = sum(1 for _ in data.rglob("*") if _.is_file()) if data.exists() else 0
    return f"hydrated {bundle} files={n}"


@app.function(image=image, timeout=12 * 3600, memory=32768, cpu=8.0, volumes={VOL: vol})
def run_cpu(model: str, split: str) -> str:
    return _run(model, split)


@app.function(
    image=image,
    gpu=["A10G", "L40S", "L4"],
    timeout=12 * 3600,
    memory=32768,
    volumes={VOL: vol},
    max_containers=6,
)
def run_gpu(model: str, split: str) -> str:
    return _run(model, split)


@app.function(image=image, timeout=600, volumes={VOL: vol})
def collect() -> str:
    root = Path(VOL)
    lines = []
    for p in sorted(root.rglob("*.csv")):
        if "results" in p.parts:
            lines.append(str(p))
    for p in sorted((root / "logs").glob("*")) if (root / "logs").exists() else []:
        lines.append(f"{p.name} {p.stat().st_size}")
    return "\n".join(lines) if lines else "empty"


@app.function(image=image, timeout=14 * 3600)
def kickoff(include_pilsl: bool = False, models: str = "") -> str:
    wanted = {m.strip() for m in models.split(",") if m.strip()}
    calls = []
    cpu = [m for m in CPU_MODELS if not wanted or m in wanted]
    gpu = [m for m in GPU_MODELS if not wanted or m in wanted]
    if include_pilsl and (not wanted or "PiLSL" in wanted):
        gpu = gpu + ["PiLSL"]
    for m in cpu:
        for s in SPLITS:
            calls.append(run_cpu.spawn(m, s))
    for m in gpu:
        for s in SPLITS:
            calls.append(run_gpu.spawn(m, s))
    return "\n".join(c.get() for c in calls)


@app.local_entrypoint()
def main(stage: str = "all", models: str = ""):
    if stage in ("all", "hydrate"):
        print(hydrate.remote("small"))
    if stage == "hydrate-large":
        print(hydrate.remote("large"))
    if stage in ("all", "run"):
        call = kickoff.spawn(False, models)
        print(f"kickoff {call.object_id}")
    if stage == "pilsl":
        call = kickoff.spawn(True, models)
        print(f"kickoff {call.object_id}")
    if stage == "collect":
        print(collect.remote())
