import {test} from 'node:test';
import assert from 'node:assert/strict';
import {JOB, PREFIX, outputKey, MAX_BYTES} from './routing.mjs';
import {featurePermissions, mintFeatureTickets, mintReadinessTickets, transfer} from './transfers.mjs';

test('readiness links exclude outer test labels and unrelated bucket data', async()=>{
  const manifest=await mintReadinessTickets('synthetic-secret','https://example.test',1700000000000);
  assert.ok(manifest.tickets.every(t=>!t.name.includes('-test-')));
  assert.ok(manifest.tickets.filter(t=>t.method==='PUT').every(t=>t.job==='readiness-r2-20260912-v1'));
  assert.ok(manifest.tickets.some(t=>t.name==='human-combinations-manifest.json'));
});

test('preparation output paths stay within one versioned SLp prefix', () => {
  assert.equal(outputKey('/outputs/complete.json'), PREFIX + 'complete.json');
  for (const path of ['/outputs/../other.json', '/outputs/%2e%2e.json', '/slp/raw/x', '/outputs/a/b.json', '/outputs/x.json?x']) {
    assert.equal(outputKey(path), null);
  }
  assert.ok(PREFIX.startsWith('slp/prepared/' + JOB));
  assert.equal(MAX_BYTES, 33554432);
});

test('feature capabilities are exact-object, method-bound and expiring', async () => {
  const now=1700000000000, secret='synthetic-test-secret';
  const manifest=await mintFeatureTickets(secret,'https://example.test',now);
  assert.equal(manifest.tickets.length,featurePermissions().length);
  assert.ok(manifest.tickets.every(p=>p.job.startsWith('identity-')||p.job.startsWith('esm-')));
  const ticket=manifest.tickets[0];
  let gets=0;
  const env={ACCESS_TOKEN:secret,CORPUS:{get:async()=>{gets++;return {size:2,body:'{}'};}}};
  assert.equal((await transfer(new Request(ticket.url),env,now)).status,200);
  assert.equal(gets,1);
  assert.equal((await transfer(new Request(ticket.url,{method:'DELETE'}),env,now)).status,403);
  assert.equal((await transfer(new Request(ticket.url),env,now+10801000)).status,403);
  assert.equal((await transfer(new Request(ticket.url+'x'),env,now)).status,403);
  assert.equal(gets,1);
});

test('upload capability cannot overwrite an existing artifact or exceed its size cap', async () => {
  const now=1700000000000,secret='synthetic-test-secret';
  const manifest=await mintFeatureTickets(secret,'https://example.test',now);
  const ticket=manifest.tickets.find(p=>p.method==='PUT');
  const headers={'Content-Length':'2','X-Content-SHA256':'a'.repeat(64)};
  const env={ACCESS_TOKEN:secret,CORPUS:{head:async()=>({size:2,customMetadata:{sha256:'b'.repeat(64)}})}};
  assert.equal((await transfer(new Request(ticket.url,{method:'PUT',headers,body:'{}'}),env,now)).status,409);
  headers['Content-Length']=String(ticket.max_bytes+1);
  assert.equal((await transfer(new Request(ticket.url,{method:'PUT',headers,body:'{}'}),env,now)).status,400);
});


test('mixed-corpus capabilities remain exact and never carry benchmark test labels', async()=>{
  const manifest=await mintReadinessTickets('synthetic-secret','https://example.test',1700000000000,true);
  assert.ok(manifest.tickets.every(t=>!t.name.includes('-test-')));
  assert.ok(manifest.tickets.filter(t=>t.method==='PUT').every(t=>t.job==='readiness-r2-20260912-v2'));
  assert.ok(manifest.tickets.some(t=>t.job==='population-yeast-r2-20260912-v4'));
  assert.ok(manifest.tickets.every(t=>t.max_bytes<=33554432));
});
