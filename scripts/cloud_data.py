"""Control cloud-to-cloud source ingestion; only JSON responses reach this host."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def settings():
    values = {}
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition('=')
            if sep and key.strip() in {'SLP_STORAGE_URL', 'SLP_STORAGE_TOKEN'}:
                values[key.strip()] = value.strip().strip('\"\'')
    for key in ('SLP_STORAGE_URL', 'SLP_STORAGE_TOKEN'):
        if os.environ.get(key):
            values[key] = os.environ[key]
    if not values.get('SLP_STORAGE_URL') or not values.get('SLP_STORAGE_TOKEN'):
        raise ValueError('Set SLP_STORAGE_URL and SLP_STORAGE_TOKEN in .env or the environment')
    if not values['SLP_STORAGE_URL'].startswith('https://'):
        raise ValueError('SLP_STORAGE_URL must use HTTPS')
    return values


def call(config, path, method='GET'):
    request = urllib.request.Request(config['SLP_STORAGE_URL'].rstrip('/') + path,
        method=method, data=b'' if method == 'POST' else None,
        headers={'Authorization': 'Bearer ' + config['SLP_STORAGE_TOKEN'],
                 'User-Agent': 'SLp-Cloud-Storage/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = response.read(1024 * 1024 + 1)
    except urllib.error.HTTPError as error:
        # The service returns bounded JSON errors, not source-file contents.
        detail = error.read(4096).decode('utf-8', errors='replace')
        raise RuntimeError(f'Cloud storage HTTP {error.code}: {detail}') from None
    if len(body) > 1024 * 1024:
        raise ValueError('Unexpectedly large control response')
    return json.loads(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'status', 'ingest'])
    parser.add_argument('ids', nargs='*', help='Source IDs; ingest defaults to the pinned manifest')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'storage/source-objects.json').read_text())
    known = {row['id']: row for row in manifest['objects']}
    if any(ident not in known for ident in args.ids):
        parser.error('Unknown source ID')
    selected = [known[ident] for ident in dict.fromkeys(args.ids)] if args.ids else list(known.values())
    if args.command == 'plan':
        total = sum(row['bytes'] for row in selected)
        print(json.dumps({'version': manifest['version'], 'objects': len(selected), 'bytes': total,
            'storage_usd_per_month_before_free_tier': round(total / 1e9 * .015, 4),
            'data_path': 'publisher -> Cloudflare Worker -> private R2',
            'local_dataset_bytes': 0, 'ids': [row['id'] for row in selected]}, indent=2))
        return
    config = settings()
    if args.command == 'status':
        print(json.dumps(call(config, '/status'), indent=2))
        return
    remote = call(config, '/manifest')
    if remote != manifest:
        raise ValueError('Deployed source manifest differs from checkout; deploy the intended version first')
    for row in selected:
        print(json.dumps(call(config, '/ingest/' + row['id'], 'POST')), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, urllib.error.URLError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
