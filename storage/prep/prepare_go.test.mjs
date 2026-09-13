import test from 'node:test';
import assert from 'node:assert/strict';
import {annotation, humanAccessions, prepareGo} from './prepare_go.mjs';

test('HGNC mapping preserves empty terminal TSV fields', () => {
  const result = humanAccessions('hgnc_id\tuniprot_ids\tunused\nHGNC:1\tP12345\t\n', new Set(['9606:HGNC:1']));
  assert.equal(result.accessions.get('P12345'), '9606:HGNC:1');
});

const human = {db: 'UniProtKB', taxon: 9606};
function row(changes={}) {
  const r = ['UniProtKB','P12345','SYMBOL','enables','GO:0003674','PMID:1','IDA','','F','name','','protein','taxon:9606','20220101','GOA','',''];
  for (const [k,v] of Object.entries(changes)) r[Number(k)] = v;
  return r;
}

test('static selection excludes negation, biological process, perturbation evidence, later dates and other taxa', () => {
  assert.equal(annotation(row(),human).term,'F:GO:0003674');
  for (const code of ['HEP','HGI','HMP','IEP','IGI','IMP']) assert.equal(annotation(row({6:code}),human).excluded,'perturbation_evidence');
  assert.equal(annotation(row({3:'NOT|enables'}),human).excluded,'negated');
  assert.equal(annotation(row({8:'P'}),human).excluded,'aspect');
  assert.equal(annotation(row({13:'20230101'}),human).excluded,'date');
  assert.equal(annotation(row({12:'taxon:10090'}),human).excluded,'identity');
  assert.throws(() => annotation(row().slice(0,16),human),/field count/);
});

test('native yeast join exposes exact systematic ORFs rather than free-text names', () => {
  const parsed=annotation(row({0:'SGD',1:'S000001',2:'ACT1',10:'ACT1|YFL039C|actin gene|Q0045',12:'taxon:559292'}),{db:'SGD',taxon:559292});
  assert.deepEqual(parsed.orfs,['YFL039C','Q0045']);
  assert.equal(parsed.object,'S000001');
});

test('gzip publication is deterministic for identical input', async () => {
  const encode=async () => new Uint8Array(await new Response(new Blob(['{"gene":"9606:A"}\n']).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer());
  assert.deepEqual(await encode(),await encode());
});

test('cloud preparation verifies inputs, joins exact accessions and publishes deterministic replayable shards', async () => {
  const encoder=new TextEncoder(), objects=new Map();
  const digest=async b => Buffer.from(await crypto.subtle.digest('SHA-256',b)).toString('hex');
  const compress=async text => new Uint8Array(await new Response(new Blob([text]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer());
  async function seed(key,body) {
    const bytes=typeof body==='string'?encoder.encode(body):body;
    objects.set(key,{bytes,sha:await digest(bytes)});
  }
  const bucket={
    async head(key) { const v=objects.get(key);return v?{size:v.bytes.length,customMetadata:{sha256:v.sha}}:null; },
    async get(key) { const v=objects.get(key);return v?{size:v.bytes.length,checksums:{sha256:Buffer.from(v.sha,'hex')},
      body:new Blob([v.bytes]).stream(),arrayBuffer:async()=>v.bytes,json:async()=>JSON.parse(new TextDecoder().decode(v.bytes))}:null; },
    async put(key,body,options) {
      if(objects.has(key))return null;
      const bytes=new Uint8Array(body);
      assert.equal(await digest(bytes),options.sha256);
      await seed(key,bytes);return {size:bytes.length};
    },
  };
  const prefix='slp/prepared/identity-r2-20260912-v5/';
  const roster=await compress([
    {gene:'9606:HGNC:1',candidate_sequences:{P12345:'hash'}},
    {gene:'9606:HGNC:2',candidate_sequences:{P99999:'hash'}},
    {gene:'9606:HGNC:3',candidate_sequences:{P99999:'hash'}},
  ].map(v=>JSON.stringify(v)).join('\n'));
  await seed(prefix+'sequence-inputs-00000.jsonl.gz',roster);
  const hgnc=new TextEncoder().encode('hgnc_id\tuniprot_ids\nHGNC:1\tP12345\nHGNC:2\tP99999\nHGNC:3\tP99999\n');
  await seed('slp/prepared/identity-r2-20260912-v3/hgnc-raw.bin.gz',await compress(new TextDecoder().decode(hgnc)));
  await seed(prefix+'sequence-inputs-manifest.json',JSON.stringify({source:{sources:{hgnc:{raw_job:'identity-r2-20260912-v3',bytes:hgnc.length,sha256:await digest(hgnc)}}},shards:[{
    name:'sequence-inputs-00000.jsonl.gz',bytes:roster.length,sha256:await digest(roster)}]}));
  const gaf=await compress([row(),row({3:'NOT|enables'}),row({1:'P99999'})].map(v=>v.join('\t')).join('\n'));
  await seed('raw.gaf.gz',gaf);
  const manifest={objects:[{id:'goa-human-20220919',key:'raw.gaf.gz',bytes:gaf.length,
    checksum:{algorithm:'sha256',value:await digest(gaf)}}]};
  const result=await prepareGo(bucket,'human',manifest,'test-source');
  assert.equal(result.rows,1);assert.equal(result.admitted_rows,1);assert.equal(result.ambiguous_accessions,1);
  assert.equal(result.excluded.negated,1);assert.equal(result.excluded.unresolved_or_ambiguous,1);
  const shard=objects.get(`slp/runs/slp-1.2-r2/${result.job}/${result.shards[0].name}`);
  const decoded=await new Response(new Blob([shard.bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
  assert.deepEqual(JSON.parse(decoded.trim()),{gene:'9606:HGNC:1',terms:['F:GO:0003674']});
  assert.deepEqual(await prepareGo(bucket,'human',manifest,'test-source'),result);
  await assert.rejects(prepareGo(bucket,'human',manifest,'changed'),/new static preparation version/);
});
