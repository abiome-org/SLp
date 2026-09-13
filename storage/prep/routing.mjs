export const JOB = 'protocol-r2-20260912-v4';
export const JOBS = new Set(['inventory-r2-20260912-v1', 'native-r2-20260912-v1', 'identity-r2-20260912-v1', 'identity-r2-20260912-v2', 'identity-r2-20260912-v3', 'identity-r2-20260912-v4', 'identity-r2-20260912-v5', 'protocol-r2-20260912-v1', 'protocol-r2-20260912-v2', 'protocol-r2-20260912-v3', JOB]);
export const PREFIX = `slp/prepared/${JOB}/`;
export const ARTIFACT_JOBS = new Set(['esm-r2-20260912-v1', 'readiness-r2-20260912-v1', 'readiness-r2-20260912-v2', 'readiness-r2-20260912-v3', 'optimizer-readiness-r2-20260912-v1']);
export const PREPARATION_JOBS = new Set(Object.keys(jobs));
export const MAX_BYTES = 32 * 1024 * 1024;
export function outputKey(path) {
  const versioned = path.match(/^\/outputs\/([a-zA-Z0-9_-]+)\/([a-zA-Z0-9_-]+\.(?:json|jsonl\.gz|bin\.gz))$/);
  if (versioned) return JOBS.has(versioned[1]) || ARTIFACT_JOBS.has(versioned[1]) || PREPARATION_JOBS.has(versioned[1]) ? `slp/prepared/${versioned[1]}/${versioned[2]}` : null;
  const match = path.match(/^\/outputs\/([a-zA-Z0-9_-]+\.(?:json|jsonl\.gz))$/);
  return match ? PREFIX + match[1] : null;
}
export const json = (value, status = 200) => Response.json(value, {
  status, headers: {'Cache-Control': 'no-store'},
});
import jobs from './prep-jobs.json' with {type: 'json'};
