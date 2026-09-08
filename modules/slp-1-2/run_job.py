"""Run the main training segment, export its result, then request pod cleanup."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from io_utils import write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--max-hours', type=float, required=True)
    p.add_argument('--cleanup-request', type=Path, required=True)
    args = p.parse_args()
    module = Path(__file__).parent
    try:
        subprocess.run([sys.executable, '-u', str(module / 'train.py'), '--root', str(args.root),
                        '--output', str(args.run), '--resume', str(args.resume), '--max-hours', str(args.max_hours)], check=True)
        subprocess.run([sys.executable, '-u', str(module / 'finish.py'), '--root', str(args.root),
                        '--run', str(args.run), '--bundle', str(args.bundle),
                        '--cleanup-request', str(args.cleanup_request)], check=True)
    finally:
        if not args.cleanup_request.exists():
            write_json(args.cleanup_request, {'status': 'job_exited_before_export', 'terminate_at': time.time() + 1800})
