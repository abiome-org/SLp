import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import worker, {ingest, verified} from './worker.mjs';
import manifest from './source-objects.json' with {type: 'json'};

const bytes = new TextEncoder().encode('experimental source');
const sha = createHash('sha256').update(bytes).digest();
const spec = {id: 'fixture', key: 'slp/raw/fixture/data', url: 'https://example.org/data',
  bytes: bytes.length, checksum: {algorithm: 'sha256', value: sha.toString('hex')}, license: 'fixture'};
const object = () => ({size: bytes.length, checksums: {sha256: sha}});

test('unauthorized callers cannot inspect manifests or issue storage operations', async () => {
  for (const route of ['/manifest', '/status', '/ingest/' + manifest.objects[0].id]) {
    const response = await worker.fetch(new Request('https://slp.test' + route), {ACCESS_TOKEN: 'secret'});
    assert.equal(response.status, 401);
  }
});

test('a caller cannot supply an arbitrary source URL or bucket key', async () => {
  const response = await worker.fetch(new Request('https://slp.test/ingest/unknown', {
    method: 'POST', headers: {Authorization: 'Bearer secret'}, body: '{"url":"https://evil.test"}',
  }), {ACCESS_TOKEN: 'secret'});
  assert.equal(response.status, 404);
});

test('all pinned objects fit the cloud streaming path and have unique immutable keys', () => {
  assert.equal(new Set(manifest.objects.map(row => row.id)).size, manifest.objects.length);
  assert.equal(new Set(manifest.objects.map(row => row.key)).size, manifest.objects.length);
  for (const row of manifest.objects) {
    assert.ok(row.bytes > 0 && row.bytes < 5 * 1024 ** 3);
    assert.ok(row.key.startsWith('slp/raw/'));
    assert.ok(row.url.startsWith('https://'));
    assert.match(row.checksum.value, row.checksum.algorithm === 'md5' ? /^[a-f0-9]{32}$/ : /^[a-f0-9]{64}$/);
  }
});

test('streaming upload pins the checksum and prevents overwrites', async () => {
  let written = false;
  const bucket = {head: async () => null, put: async (key, stream, options) => {
    assert.equal(key, spec.key);
    assert.equal(options.onlyIf.get('If-None-Match'), '*');
    assert.equal(options.sha256, spec.checksum.value);
    assert.equal(options.customMetadata.storage_role, 'source-only');
    assert.deepEqual(new Uint8Array(await new Response(stream).arrayBuffer()), bytes);
    written = true;
    return object();
  }};
  const response = await ingest(spec, bucket, async () => new Response(bytes, {headers: {'Content-Length': String(bytes.length)}}));
  assert.equal(response.status, 200);
  assert.ok(written);
});

test('small objects tolerate a missing or unusable publisher length header', async () => {
  let written = false;
  const bucket = {head: async () => null, put: async (key, body) => {
    assert.deepEqual(new Uint8Array(body), bytes);
    written = true;
    return object();
  }};
  const response = await ingest(spec, bucket, async () => new Response(bytes));
  assert.equal(response.status, 200);
  assert.ok(written);
});

test('publisher byte drift fails before any R2 write', async () => {
  const bucket = {head: async () => null, put: async () => assert.fail('must not write')};
  const drifted = new TextEncoder().encode('drifted');
  const response = await ingest(spec, bucket, async () => new Response(drifted,
    {headers: {'Content-Length': String(drifted.length)}}));
  assert.equal(response.status, 502);
});

test('large objects use a fixed-length stream when the publisher length is unusable', async () => {
  const largeSpec = {...spec, bytes: 4 * 1024 ** 2 + 1};
  let expectedLength = null;
  class FakeFixedLengthStream {
    constructor(length) {
      expectedLength = length;
      return new TransformStream();
    }
  }
  const bucket = {head: async () => null, put: async (key, body) => {
    assert.deepEqual(new Uint8Array(await new Response(body).arrayBuffer()), bytes);
    return {...object(), size: largeSpec.bytes};
  }};
  const response = await ingest(largeSpec, bucket, async () => new Response(bytes), FakeFixedLengthStream);
  assert.equal(response.status, 200);
  assert.equal(expectedLength, largeSpec.bytes);
});

test('verified retries skip upstream; corrupted existing objects are never overwritten', async () => {
  const fetcher = async () => assert.fail('must not fetch');
  const good = await ingest(spec, {head: async () => object()}, fetcher);
  assert.equal((await good.json()).state, 'verified-existing');
  const bad = await ingest(spec, {head: async () => ({...object(), size: 1})}, fetcher);
  assert.equal(bad.status, 409);
  assert.ok(!verified({size: bytes.length, checksums: {}}, spec));
});

test('a concurrent winner must pass the same checksum check', async () => {
  let heads = 0;
  const bucket = {head: async () => ++heads === 1 ? null : object(), put: async () => null};
  const response = await ingest(spec, bucket, async () => new Response(bytes, {headers: {'Content-Length': String(bytes.length)}}));
  assert.equal((await response.json()).state, 'verified-existing');
});
