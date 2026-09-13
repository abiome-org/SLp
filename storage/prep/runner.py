"""One fixed cloud job per process; no user-supplied commands or file paths."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import signal
import subprocess
import threading
import traceback
from pathlib import Path

lock = threading.Lock()
started = False
BUILD = Path("build.sha256").read_text().strip()
JOB = os.environ["SLP_JOB"]
ENTRY = (
    "prepare_protocol.py"
    if JOB.startswith("protocol-")
    else "pack_corpus.py"
    if JOB.startswith("pack-")
    else "prepare_identity.py"
    if JOB.startswith("identity-")
    else None
)
JOBS = json.loads(Path("prep-jobs.json").read_text())


def execute(job):
    try:
        entry = JOBS[job]["entry"] if job in JOBS else ENTRY if job == JOB else None
        if entry is None:
            raise ValueError("Unknown fixed preparation job")
        subprocess.run(
            ["python", entry],
            env={**os.environ, "SLP_JOB": job},
            check=True,
            timeout=1740,
        )
    except Exception as exc:
        from inspect_corpus import request

        body = json.dumps(
            {
                "state": "failed",
                "type": type(exc).__name__,
                "job": job,
                "returncode": getattr(exc, "returncode", None),
                "timeout_seconds": getattr(exc, "timeout", None),
            }
        ).encode()
        try:
            with request(
                f"/outputs/{job}/runner-failed.json", data=body, prep=True
            ) as response:
                response.read(4096)
        except Exception:
            traceback.print_exc()
    finally:
        # Terminate the container process even if its client disconnects.
        os._exit(0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        body = json.dumps({"build_sha256": BUILD, "entry": ENTRY, "job": JOB}).encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        global started
        job = self.headers.get("X-SLP-Job", JOB)
        with lock:
            if (
                self.path != "/run"
                or started
                or self.headers.get("X-SLP-Build") != BUILD
                or (job not in JOBS and job != JOB)
            ):
                self.send_response(409)
                self.end_headers()
                return
            started = True
        body = json.dumps(
            {"state": "started", "job": job, "maximum_seconds": 1800}
        ).encode()
        self.send_response(202)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        threading.Thread(target=execute, args=(job,), daemon=True).start()


if __name__ == "__main__":
    # Python is PID 1 here; install a handler rather than relying on the Linux
    # default SIGTERM disposition. Active jobs are never intentionally redeployed.
    signal.signal(signal.SIGTERM, lambda *_: os._exit(143))
    timer = threading.Timer(1800, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
