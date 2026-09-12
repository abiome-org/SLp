"""Independent in-pod wall-clock guard. Uses the pod's built-in scoped key.

Only the pod named by RUNPOD_POD_ID can be deleted. Checkpoints must live on a
network volume. The account-wide API key is never needed on the training pod.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error


def main(args):
    # Non-login SSH does not inherit the container's injected environment.
    # Read only the two provider-injected values from init; never print the key.
    init = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    pod_id = os.environ.get('RUNPOD_POD_ID') or init[b'RUNPOD_POD_ID'].decode()
    key = os.environ.get('RUNPOD_API_KEY') or init[b'RUNPOD_API_KEY'].decode()
    # The injected pod-scoped key supports GraphQL but returns 403 on REST v1.
    # Use the provider's native self-termination mutation with header auth.
    def request(query):
        req = urllib.request.Request('https://api.runpod.io/graphql',
              data=json.dumps({'query': query}).encode(),
              headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                       'User-Agent': 'runpod-cli/2.12.0 (SLp campaign guard)'})
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.load(response)
        if result.get('errors'):
            raise RuntimeError('Provider rejected scoped pod operation')
        return result['data']
    identity = request('query { pod(input: {podId: ' + json.dumps(pod_id) + '}) { id } }')
    if identity['pod']['id'] != pod_id:
        raise RuntimeError('Scoped credential identifies a different pod')
    print(json.dumps({'event': 'pod_guard_armed', 'pod_id': pod_id, 'terminate_at': args.deadline,
                      'scoped_credential_verified': True}), flush=True)
    while time.time() < args.deadline:
        if args.cleanup_request and args.cleanup_request.exists():
            try:
                cleanup = json.loads(args.cleanup_request.read_text())
                args.deadline = min(args.deadline, float(cleanup['terminate_at']))
            except (ValueError, KeyError):
                pass  # A malformed request cannot disable the hard deadline.
        if time.time() >= args.deadline:
            break
        time.sleep(max(0, min(30, args.deadline - time.time())))
    while True:
        try:
            request('mutation { podTerminate(input: {podId: ' + json.dumps(pod_id) + '}) }')
            print(json.dumps({'event': 'self_terminate', 'submitted': True}), flush=True)
            return
        except Exception as error:
            print(json.dumps({'event': 'self_terminate_retry', 'error_type': type(error).__name__}), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--cleanup-request', type=Path)
    main(parser.parse_args())
