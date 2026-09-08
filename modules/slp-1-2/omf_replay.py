"""Expose the bundle's CPU replay in OMF's ordinary-script result protocol."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from io_utils import write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = subprocess.run([sys.executable, str(args.bundle / 'replay.py'), '--bundle', str(args.bundle)],
             check=True, capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    report = json.loads(result.stdout.splitlines()[-1])
    errors = report['maximum_absolute_errors']
    compatible = report['replay'] == 'passed' and set(errors) == {'mean', 'log_variance', 'sample'}
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'metrics.json', {'passed': compatible, 'compatibilityPassed': compatible,
               'portability_max_error': max(errors.values()), 'optimization_steps': 0})
    if not compatible:
        raise AssertionError('Standalone replay did not satisfy its contract')
