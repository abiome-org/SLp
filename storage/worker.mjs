// Dataset bytes travel from the publisher to Cloudflare, never through the caller.
import manifest from './source-objects.json' with { type: 'json' };

const objects = new Map(manifest.objects.map(row => [row.id, row]));
const json = (body, status = 200) => Response.json(body, {
  status, headers: { 'Cache-Control': 'no-store' },
});
const hex = buffer => Array.from(new Uint8Array(buffer), n => n.toString(16).padStart(2, '0')).join('');

export function verified(object, spec) {
  const checksum = object?.checksums?.[spec.checksum.algorithm];
  return object?.size === spec.bytes && checksum && hex(checksum) === spec.checksum.value;
}

export async function ingest(spec, bucket, fetcher = fetch) {
  const existing = await bucket.head(spec.key);
  if (existing) {
    if (!verified(existing, spec)) return json({error: 'Existing object fails checksum or size', id: spec.id}, 409);
    return json({id: spec.id, key: spec.key, bytes: existing.size, state: 'verified-existing'});
  }
  if (spec.bytes > 5 * 1024 ** 3) return json({error: 'Use a cloud CPU multipart importer for objects above 5 GiB'}, 422);
  const upstream = await fetcher(spec.url, {headers: {'Accept-Encoding': 'identity', 'User-Agent': 'SLp-Cloud-Storage/1.0'}});
  if (upstream.status !== 200 || !upstream.body) {
    await upstream.body?.cancel();
    return json({error: 'Publisher download failed', upstream_status: upstream.status, id: spec.id}, 502);
  }
  // Streaming R2 writes require a known-length body. Fail rather than buffer a file.
  if (Number(upstream.headers.get('Content-Length')) !== spec.bytes) {
    await upstream.body.cancel();
    return json({error: 'Publisher size differs from pinned manifest', id: spec.id}, 502);
  }
  const result = await bucket.put(spec.key, upstream.body, {
    onlyIf: new Headers({'If-None-Match': '*'}),
    [spec.checksum.algorithm]: spec.checksum.value,
    storageClass: 'Standard',
    httpMetadata: {contentType: upstream.headers.get('Content-Type') || 'application/octet-stream'},
    customMetadata: {source_id: spec.id, source_url: spec.url, manifest_version: manifest.version,
      storage_role: 'source-only', license: spec.license},
  });
  // A competing identical import may have won the conditional write.
  const stored = result || await bucket.head(spec.key);
  if (!verified(stored, spec)) return json({error: 'Stored object verification failed', id: spec.id}, 409);
  return json({id: spec.id, key: spec.key, bytes: stored.size, state: result ? 'verified-uploaded' : 'verified-existing'});
}

export default {
  async fetch(request, env) {
    if (!env.ACCESS_TOKEN || request.headers.get('Authorization') !== `Bearer ${env.ACCESS_TOKEN}`) {
      return json({error: 'Unauthorized'}, 401);
    }
    const pathname = new URL(request.url).pathname;
    if (request.method === 'GET' && pathname === '/manifest') return json(manifest);
    if (request.method === 'GET' && pathname === '/status') {
      const rows = [];
      for (const spec of objects.values()) {
        const stored = await env.CORPUS.head(spec.key);
        rows.push({id: spec.id, key: spec.key, bytes: spec.bytes,
          state: !stored ? 'missing' : verified(stored, spec) ? 'verified' : 'mismatch'});
      }
      return json({version: manifest.version, objects: rows});
    }
    const match = pathname.match(/^\/(ingest|objects)\/([a-zA-Z0-9_-]+)$/);
    const spec = match && objects.get(match[2]);
    if (!spec) return json({error: 'Unknown source object'}, 404);
    try {
      if (request.method === 'POST' && match[1] === 'ingest') return await ingest(spec, env.CORPUS);
      if (['GET', 'HEAD'].includes(request.method) && match[1] === 'objects') {
        const stored = await env.CORPUS.head(spec.key);
        if (!stored) return json({error: 'Object not imported'}, 404);
        if (!verified(stored, spec)) return json({error: 'Object fails manifest verification'}, 409);
        const headers = new Headers({'Content-Length': String(stored.size),
          'ETag': stored.httpEtag, 'Cache-Control': 'private, no-store'});
        stored.writeHttpMetadata(headers);
        if (request.method === 'HEAD') return new Response(null, {headers});
        const object = await env.CORPUS.get(spec.key);
        if (!object || object.etag !== stored.etag) return json({error: 'Object changed while reading'}, 409);
        return new Response(object.body, {headers});
      }
      return json({error: 'Method not allowed'}, 405);
    } catch (error) {
      // Do not expose credentials, signed redirect URLs or upstream response bodies.
      console.error('SLp source operation failed', spec.id, error.name);
      return json({error: 'Cloud transfer failed; retry the same source ID', id: spec.id}, 502);
    }
  },
};
