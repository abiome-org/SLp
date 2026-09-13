"""Execute a frozen nested-CV3 packet only with explicit research authorization.

Run alongside the local ticket broker. Each completed stage is verified in R2
before its fold's local checkpoint cache may be removed. No official test data
is opened until that fold's inner selection and fresh outer refit are complete.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import torch

from cloud_io import upload_directory, download_directory, get_json, fetch_shards
from data import digest
from evaluate import select_checkpoint


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True))
    tmp.replace(path)


def outer_recipe(recipe, checkpoint_update):
    result = json.loads(json.dumps(recipe))
    if checkpoint_update is not None:
        # A 2k probe was sampled from the 8k learning-rate schedule. Preserve
        # that schedule during refitting, stopping at 2k without compressing it.
        result["pretrain"]["stop_at_update"] = checkpoint_update
        result["pretrain"]["retain_updates"] = []
    return result


def choose(reports, fold, expected_candidates):
    if {r["checkpoint"] for r in reports} != set(expected_candidates):
        raise ValueError("Incomplete candidate comparison")
    return select_checkpoint(
        reports, [fold["benchmark"]], {fold["benchmark"]: [fold["name"]]}
    )


def validate_job_matrix(plan):
    folds = [fold["name"] for fold in plan["protocols"]]
    variants = set(plan["variants"])
    actual = [(job["fold"], job["candidate"]) for job in plan["inner_jobs"]]
    expected = {(fold, variant) for fold in folds for variant in variants}
    if (
        len(folds) != 30 or len(set(folds)) != 30 or not variants
        or len(actual) != len(set(actual)) or set(actual) != expected
    ):
        raise ValueError("The declared suite is incomplete or duplicated")
    if plan.get("outer_evaluation", "selected") not in ("selected", "all_families"):
        raise ValueError("Unknown outer evaluation policy")


class Runner:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root).resolve()
        self.packet = Path(args.plan).resolve().parent
        self.plan = json.loads(Path(args.plan).read_text())
        self.state = self.root / "campaign-state"
        self.queue = self.root / "transfer-requests"
        self.queue.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(exist_ok=True)
        self.journal = self.state / "journal.json"
        self.saved = (
            json.loads(self.journal.read_text())
            if self.journal.exists()
            else {"plan_sha256": digest(args.plan), "stages": {}, "folds": {}}
        )
        if self.saved["plan_sha256"] != digest(args.plan):
            raise ValueError("Cannot resume a changed campaign packet")

    def validate(self):
        if self.plan["schema"] != "slp.r2-training-packet/v1":
            raise ValueError("Unknown campaign schema")
        for relative, sha in self.plan["source"].items():
            if digest(Path(__file__).parent / Path(relative).name) != sha:
                raise ValueError("Captured model source changed: " + relative)
        for spec in self.plan["variants"].values():
            if digest(self.packet / spec["recipe"]) != spec["sha256"]:
                raise ValueError("Candidate recipe changed")
        validate_job_matrix(self.plan)
        if self.plan.get("features_sha256") and digest(
            self.root / "features/manifest.json"
        ) != self.plan["features_sha256"]:
            raise ValueError("Static feature assembly changed")
        for fold in self.plan["protocols"]:
            if (
                digest(self.root / "protocols" / fold["file"])
                != fold["metadata_sha256"]
            ):
                raise ValueError("Official protocol metadata changed")
        if digest(self.root / "mixed-corpus.json") != self.plan["corpus_sha256"]:
            raise ValueError("Fitting corpus changed")

    def command(self, argv):
        result = [str(x).replace("/workspace/slp-r2", str(self.root)) for x in argv]
        result[0] = sys.executable
        if "--deadline" in result:
            result[result.index("--deadline") + 1] = str(self.args.deadline)
        return result

    def execute(self, argv, log):
        remaining = self.args.deadline - time.time() - 600
        if remaining < 120:
            raise TimeoutError("Campaign allocation reserved for finalization")
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with Path(log).open("a") as file:
            subprocess.run(
                self.command(argv),
                cwd=Path(__file__).parent,
                stdout=file,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=remaining,
            )

    def inputs(self, fold, partition):
        spec = fold["partition_manifests"][partition]
        directory = self.root / "protocols"
        if (directory / spec["name"]).exists():
            if json.loads((directory / spec["name"]).read_text()) != spec["manifest"]:
                raise ValueError("Changed benchmark partition manifest")
            return
        job = (
            "campaign-r2-"
            + self.args.run_id
            + "-"
            + fold["name"].replace("_", "-")
            + "-input-"
            + partition
        )
        request = self.queue / (job + ".json")
        ready = self.queue / (job + ".ready.json")
        write_json(
            request,
            {
                "job": job,
                "kind": "partition",
                "plan_sha256": self.saved["plan_sha256"],
                "fold": fold["name"],
                "partition": partition,
            },
        )
        while not ready.exists():
            if self.args.deadline - time.time() < 600:
                raise TimeoutError("Input ticket broker did not respond")
            time.sleep(2)
        os.environ["SLP_TRANSFER_TICKETS"] = str(ready)
        if get_json(fold["job"], spec["name"]) != spec["manifest"]:
            raise ValueError("Remote official partition manifest changed")
        fetch_shards(fold["job"], spec["name"], directory)
        request.unlink()

    def publish(self, directory, label):
        job = "campaign-r2-" + self.args.run_id + "-" + label.replace("_", "-")
        request = self.queue / (job + ".json")
        ready = self.queue / (job + ".ready.json")
        with tempfile.TemporaryDirectory(
            dir=directory.parent, prefix=".publish-"
        ) as temporary:
            snapshot = Path(temporary)
            seen, aliases = {}, {}
            for path in sorted(
                directory.iterdir(), key=lambda p: (p.name != "checkpoint.pt", p.name)
            ):
                if not path.is_file() or path.suffix in (".log", ".tmp"):
                    continue
                inode = (path.stat().st_dev, path.stat().st_ino)
                if inode in seen:
                    aliases[path.name] = seen[inode]
                    continue
                seen[inode] = path.name
                os.link(path, snapshot / path.name)
            files = {
                p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                for p in snapshot.iterdir()
            }
            published = self.saved.setdefault("published_jobs", {})
            ceiling = (
                self.plan.get("cost_estimate", {}).get(
                    "r2_retained_compressed_gb_ceiling", 600
                )
                * 1_000_000_000
            )
            # Include a bound above DEFLATE's possible expansion and gzip headers.
            archive_upper = sum(
                f["bytes"]
                + f["bytes"] // 1000
                + 64 * ((f["bytes"] + 16 * 1024**2 - 1) // (16 * 1024**2))
                for f in files.values()
            )
            if (
                job not in published
                and sum(published.values()) + archive_upper > ceiling
            ):
                raise ValueError("Declared R2 archive allocation is exhausted")
            write_json(
                request,
                {"job": job, "plan_sha256": self.saved["plan_sha256"], "files": files},
            )
            while not ready.exists():
                if self.args.deadline - time.time() < 300:
                    raise TimeoutError("Publication ticket broker did not respond")
                time.sleep(2)
            permission = json.loads(ready.read_text())
            if permission.get("job") != job or permission.get("files") != files:
                raise ValueError("Publication capability has mismatched files")
            os.environ["SLP_TRANSFER_TICKETS"] = str(ready)
            manifest = upload_directory(snapshot, job)
            remote = get_json(job, "artifact.json")
            if remote != manifest:
                raise ValueError("Published stage manifest was not readable")
            published[job] = sum(
                p["bytes"] for f in manifest["files"].values() for p in f["parts"]
            )
            write_json(self.journal, self.saved)
        request.unlink()
        return {"job": job, "files": files, "aliases": aliases}

    def restore(self, directory, receipt):
        job = receipt["job"]
        request = self.queue / (job + ".json")
        ready = self.queue / (job + ".ready.json")
        ready.unlink(missing_ok=True)
        write_json(
            request,
            {
                "job": job,
                "kind": "restore",
                "files": receipt["files"],
                "plan_sha256": self.saved["plan_sha256"],
            },
        )
        while not ready.exists():
            if self.args.deadline - time.time() < 600:
                raise TimeoutError("Restore ticket broker did not respond")
            time.sleep(2)
        permission = json.loads(ready.read_text())
        if (
            permission.get("kind") != "restore"
            or permission.get("files") != receipt["files"]
        ):
            raise ValueError("Restore capability has mismatched files")
        os.environ["SLP_TRANSFER_TICKETS"] = str(ready)
        manifest = download_directory(job, directory)
        if {
            name: {"bytes": f["bytes"], "sha256": f["sha256"]}
            for name, f in manifest["files"].items()
        } != receipt["files"]:
            raise ValueError("Restored files differ from the campaign journal")
        for name, target in receipt.get("aliases", {}).items():
            if Path(name).name != name or target not in manifest["files"]:
                raise ValueError("Invalid retained-checkpoint alias")
            os.link(directory / target, directory / name)
        request.unlink()

    def fit(self, argv, key):
        command = self.command(argv)
        directory = Path(command[command.index("--output") + 1])
        if key in self.saved["stages"]:
            if not (directory / "checkpoint.pt").exists():
                self.restore(directory, self.saved["stages"][key])
            return directory
        if not (directory / "checkpoint.pt").exists() and key in self.saved.get(
            "partial_stages", {}
        ):
            self.restore(directory, self.saved["partial_stages"][key])
        recipe = json.loads(Path(command[command.index("--recipe") + 1]).read_text())
        stage = command[command.index("--stage") + 1]
        end = recipe[stage].get("stop_at_update", recipe[stage]["updates"])
        complete = directory / "complete.json"
        report = json.loads(complete.read_text()) if complete.exists() else None
        if not report or report["result"]["completed_update"] != end:
            if (directory / "checkpoint.pt").exists():
                if "--initialize" in command:
                    at = command.index("--initialize")
                    del command[at : at + 2]
                command += ["--resume", str(directory / "checkpoint.pt")]
            try:
                self.execute(command, directory / "fit.log")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if (directory / "checkpoint.pt").exists():
                    at = torch.load(
                        directory / "checkpoint.pt",
                        map_location="cpu",
                        weights_only=True,
                        mmap=True,
                    )["update"]
                    self.saved.setdefault("partial_stages", {})[key] = self.publish(
                        directory, key + f"-u{at:06d}"
                    )
                    write_json(self.journal, self.saved)
                raise
            report = json.loads(complete.read_text())
        if report["result"]["completed_update"] != end:
            at = report["result"]["completed_update"]
            self.saved.setdefault("partial_stages", {})[key] = self.publish(
                directory, key + f"-u{at:06d}"
            )
            write_json(self.journal, self.saved)
            raise ValueError("Incomplete fitting stage; keep its resumable checkpoint")
        self.saved["stages"][key] = self.publish(directory, key + f"-u{end:06d}")
        write_json(self.journal, self.saved)
        return directory

    def score(self, argv):
        command = self.command(argv)
        output = Path(command[command.index("--output") + 1])
        metrics = output / "metrics.json"
        if not metrics.exists():
            if output.exists():
                raise ValueError("Incomplete score directory requires inspection")
            self.execute(command, output.parent / (output.name + ".log"))
        return json.loads(metrics.read_text())

    def refit_outer(self, job, probe, run, label):
        recipe = outer_recipe(
            json.loads((self.packet / self.plan["variants"][job["candidate"]]["recipe"]).read_text()),
            probe["checkpoint_update"],
        )
        recipe_path = self.state / (label + "-recipe.json")
        write_json(recipe_path, recipe)
        command = self.command(job["fitting_command"])
        for flag, value in (("--scope", "outer"), ("--recipe", str(recipe_path))):
            command[command.index(flag) + 1] = value
        if job["pretraining_updates"]:
            command[command.index("--output") + 1] = str(run / "pretrain")
            self.fit(command, label + "-pretrain")
            command += ["--initialize", str(run / "pretrain/checkpoint.pt")]
        command[command.index("--stage") + 1] = "adapt"
        command[command.index("--output") + 1] = str(run / "adapt")
        self.fit(command, label + "-adapt")

    def score_outer(self, probe, run, label):
        score = self.command(probe["scoring_command"])
        for flag, value in (
            ("--checkpoint", str(run / "adapt/checkpoint.pt")),
            ("--scope", "outer"),
            ("--basal", str(run / "adapt/basal.json")),
            ("--output", str(run / "scores")),
        ):
            score[score.index(flag) + 1] = value
        score += ["--allow-outer-test", "--export", str(run / "bundle")]
        metrics = self.score(score)
        outputs = {
            "scores": self.publish(run / "scores", label + "-scores"),
            "bundle": self.publish(run / "bundle", label + "-bundle"),
        }
        return metrics, outputs

    def run(self):
        self.validate()
        if not self.args.execute_research:
            return {"state": "validated", "training_launched": False}
        if not self.args.allow_outer_test:
            raise ValueError(
                "A complete research campaign needs explicit final outer scoring authorization"
            )
        if "source" not in self.saved:
            captured = self.root / "captured-source"
            captured.mkdir(exist_ok=True)
            for relative in self.plan.get("source", {}):
                path = Path(__file__).parent / Path(relative).name
                if (
                    path.suffix not in (".py", ".lock", ".in")
                    and path.name != "CONTRACT.md"
                ):
                    raise ValueError("Unexpected file in captured source allowlist")
                shutil.copyfile(path, captured / path.name)
            shutil.copyfile(self.args.plan, captured / "campaign-plan.json")
            for variant, spec in self.plan["variants"].items():
                shutil.copyfile(
                    self.packet / spec["recipe"],
                    captured / ("recipe-" + variant + ".json"),
                )
            for fold in self.plan["protocols"]:
                shutil.copyfile(
                    self.root / "protocols" / fold["file"],
                    captured / ("protocol-" + fold["file"]),
                )
            self.saved["source"] = self.publish(captured, "source")
            write_json(self.journal, self.saved)
        for fold in self.plan["protocols"]:
            name = fold["name"]
            if name in self.saved["folds"]:
                continue
            jobs = [j for j in self.plan["inner_jobs"] if j["fold"] == name]
            for partition in ("train", "valid"):
                if partition in fold["partition_manifests"]:
                    self.inputs(fold, partition)
            reports, probes = [], {}
            for job in jobs:
                key = name + "-" + job["candidate"]
                self.fit(
                    job["fitting_command"],
                    key + ("-pretrain" if job["pretraining_updates"] else "-adapt"),
                )
                for probe in job["probes"]:
                    command = probe["scoring_command"]
                    candidate = command[command.index("--candidate") + 1]
                    if probe["adaptation_command"]:
                        self.fit(
                            probe["adaptation_command"],
                            name + "-" + candidate + "-adapt",
                        )
                    reports.append(self.score(command))
                    score_key = name + "-" + candidate + "-inner-scores"
                    if score_key not in self.saved["stages"]:
                        score_dir = Path(
                            self.command(command)[command.index("--output") + 1]
                        )
                        self.saved["stages"][score_key] = self.publish(
                            score_dir, score_key
                        )
                        write_json(self.journal, self.saved)
                    probes[candidate] = (job, probe)
            selection = choose(reports, fold, probes)
            all_families = self.plan.get("outer_evaluation", "selected") == "all_families"
            choices = [selection]
            if all_families:
                choices = []
                for family in self.plan["variants"]:
                    keys = {key for key, (job, _) in probes.items() if job["candidate"] == family}
                    choices.append(choose([r for r in reports if r["checkpoint"] in keys], fold, keys))
            pending = []
            for choice in choices:
                job, probe = probes[choice["selected"]]
                run = self.root / "runs" / name / "outer"
                label = name + "-outer"
                if all_families:
                    run = run / job["candidate"]
                    label += "-" + job["candidate"]
                self.refit_outer(job, probe, run, label)
                pending.append((job["candidate"], choice, probe, run, label))
            # All compared families are selected and freshly fitted before any
            # outer labels are fetched. Test outcomes cannot alter these choices.
            self.inputs(fold, "test")
            families = {}
            for family, choice, probe, run, label in pending:
                metrics, outputs = self.score_outer(probe, run, label)
                families[family] = {"selection": choice, "metrics": metrics, "outputs": outputs}
            selected_family = probes[selection["selected"]][0]["candidate"]
            selected = families[selected_family]
            selection_dir = self.root / "runs" / name / "selection"
            write_json(
                selection_dir / "selection.json",
                {"selection": selection, "inner_reports": reports,
                 "family_selections": {family: value["selection"] for family, value in families.items()}},
            )
            outputs = dict(selected["outputs"])
            outputs["selection"] = self.publish(selection_dir, name + "-selection")
            self.saved["folds"][name] = {
                "selection": selection,
                "metrics": selected["metrics"],
                "outputs": outputs,
                "families": families,
            }
            write_json(self.journal, self.saved)
            # Only this completed fold's owned cache is removed, after R2 verification.
            shutil.rmtree(self.root / "runs" / name)
        return {"state": "complete", "folds": len(self.saved["folds"])}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--root", default="/workspace/slp-r2")
    p.add_argument("--run-id", required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--execute-research", action="store_true")
    p.add_argument("--allow-outer-test", action="store_true")
    print(json.dumps(Runner(p.parse_args()).run()))
