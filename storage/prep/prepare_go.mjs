// Static functional annotations, prepared on Cloudflare without outcome data.
const encoder = new TextEncoder();
const hex = value => Array.from(new Uint8Array(value), x => x.toString(16).padStart(2, '0')).join('');
const sha = async value => hex(await crypto.subtle.digest('SHA-256', value));
const excludedEvidence = new Set(['HEP', 'HGI', 'HMP', 'IEP', 'IGI', 'IMP']);
const identityKey = 'slp/prepared/identity-r2-20260912-v5/sequence-inputs-manifest.json';
const sources = {
  human: {id: 'goa-human-20220919', taxon: 9606, db: 'UniProtKB'},
  yeast: {id: 'go-sgd-20220919', taxon: 559292, db: 'SGD'},
};

export function humanAccessions(text, genes) {
  // Empty terminal columns are data; trimming whitespace would remove them.
  const rows = text.replace(/^\uFEFF/, '').replace(/[\r\n]+$/, '').split('\n');
  const header = rows.shift().replace(/\r$/, '').split('\t');
  const geneColumn = header.indexOf('hgnc_id'), accessionColumn = header.indexOf('uniprot_ids');
  if (geneColumn < 0 || accessionColumn < 0) throw Error('Missing HGNC stable identifier columns');
  const accessions = new Map(), ambiguous = new Set();
  const value = text => text.replace(/^"|"$/g, '').replaceAll('""', '"');
  for (const [index, line] of rows.entries()) {
    const fields = line.replace(/\r$/, '').split('\t');
    if (fields.length !== header.length) throw Error(`Ambiguous HGNC TSV columns at row ${index + 2}: ${fields.length} versus ${header.length}`);
    const gene = `9606:${value(fields[geneColumn])}`;
    if (!genes.has(gene)) continue;
    for (const accession of value(fields[accessionColumn]).split('|').filter(Boolean)) {
      if (accessions.has(accession) && accessions.get(accession) !== gene) ambiguous.add(accession);
      else accessions.set(accession, gene);
    }
  }
  return {accessions, ambiguous};
}

export function annotation(fields, source) {
  if (fields.length !== 17) throw Error('Invalid GAF field count');
  if (fields[0] !== source.db || fields[12].split('|')[0] !== `taxon:${source.taxon}`) return {excluded: 'identity'};
  if (fields[3].split('|').includes('NOT')) return {excluded: 'negated'};
  if (!['F', 'C'].includes(fields[8])) return {excluded: 'aspect'};
  if (excludedEvidence.has(fields[6])) return {excluded: 'perturbation_evidence'};
  if (!/^\d{8}$/.test(fields[13]) || fields[13] > '20221231') return {excluded: 'date'};
  if (!/^GO:\d{7}$/.test(fields[4])) throw Error('Invalid GO term');
  return {object: fields[1], term: `${fields[8]}:${fields[4]}`, evidence: fields[6],
    // Only exact registered systematic ORFs are eligible for the yeast join.
    orfs: [fields[2], ...fields[10].split('|')].filter(v => /^(?:Y[A-P][LR]\d{3}[CW](?:-[A-Z])?|Q\d{4})$/.test(v))};
}

async function* lines(stream) {
  const reader = stream.pipeThrough(new DecompressionStream('gzip')).pipeThrough(new TextDecoderStream()).getReader();
  let pending = '';
  try {
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      pending += value;
      const split = pending.split('\n');
      pending = split.pop();
      if (pending.length > 8 * 1024 * 1024) throw Error('Oversized input record');
      for (const line of split) if (line.trim()) yield line.replace(/\r$/, '');
    }
    if (pending.trim()) yield pending.replace(/\r$/, '');
  } finally { reader.releaseLock(); }
}

async function immutable(bucket, key, body, digest) {
  const old = await bucket.head(key);
  if (old) {
    if (old.size !== body.byteLength || old.customMetadata?.sha256 !== digest) throw Error('Immutable annotation conflict');
    return;
  }
  const result = await bucket.put(key, body, {onlyIf: new Headers({'If-None-Match': '*'}), sha256: digest,
    customMetadata: {sha256: digest, role: 'static-functional-annotations'}});
  if (!result) {
    const raced = await bucket.head(key);
    if (raced?.size !== body.byteLength || raced.customMetadata?.sha256 !== digest) throw Error('Annotation publication race');
  }
}

export async function prepareGo(bucket, name, manifest, sourceSha) {
  const selected = sources[name];
  if (!selected) throw Error('Unknown static annotation source');
  const source = manifest.objects.find(x => x.id === selected.id);
  if (!source || source.checksum.algorithm !== 'sha256') throw Error('Unpinned annotation source');
  const job = `campaign-r2-static-go-20220919-${name}-v2`;
  const prefix = `slp/runs/slp-1.2-r2/${job}/`;
  const existing = await bucket.get(prefix + 'annotations-manifest.json');
  if (existing) {
    const saved = await existing.json();
    if (saved.source_module_sha256 !== sourceSha ||
        saved.source?.checksum?.value !== source.checksum.value) throw Error('Use a new static preparation version');
    return saved;
  }
  const identity = await bucket.get(identityKey);
  if (!identity) throw Error('Missing static gene identity');
  const identityBytes = await identity.arrayBuffer();
  const identityManifestSha = await sha(identityBytes);
  const roster = JSON.parse(new TextDecoder().decode(identityBytes));
  const genes = new Map();
  for (const shard of roster.shards) {
    if (!/^sequence-inputs-\d{5}\.jsonl\.gz$/.test(shard.name)) throw Error('Invalid roster shard');
    const object = await bucket.get(identityKey.replace('sequence-inputs-manifest.json', shard.name));
    if (!object || object.size !== shard.bytes || hex(object.checksums.sha256) !== shard.sha256) throw Error('Roster checksum mismatch');
    for await (const line of lines(object.body)) {
      const row = JSON.parse(line);
      if (!row.gene.startsWith(`${selected.taxon}:`)) continue;
      genes.set(row.gene, new Set());
    }
  }
  let accessions = new Map(), ambiguous = new Set(), mappingSource = null;
  if (name === 'human') {
    const hgnc = roster.source?.sources?.hgnc;
    if (!hgnc || hgnc.bytes > 40 * 1024 * 1024 || hgnc.raw_job !== 'identity-r2-20260912-v3') throw Error('Unpinned HGNC mapping');
    const key = `slp/prepared/${hgnc.raw_job}/hgnc-raw.bin.gz`;
    const compressed = await bucket.get(key);
    if (!compressed) throw Error('Missing captured HGNC mapping');
    const bytes = await new Response(compressed.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    if (bytes.byteLength !== hgnc.bytes || await sha(bytes) !== hgnc.sha256) throw Error('HGNC mapping checksum mismatch');
    ({accessions, ambiguous} = humanAccessions(new TextDecoder().decode(bytes), genes));
    mappingSource = {key, ...hgnc, join: 'exact HGNC uniprot_ids; ambiguous accessions excluded; no symbol join'};
  }
  const object = await bucket.get(source.key);
  if (!object || object.size !== source.bytes || hex(object.checksums.sha256) !== source.checksum.value) throw Error('GAF checksum mismatch');
  const excluded = {}, evidence = {};
  let inputRows = 0, admittedRows = 0;
  for await (const line of lines(object.body)) {
    if (line.startsWith('!')) continue;
    inputRows++;
    const parsed = annotation(line.split('\t'), selected);
    if (parsed.excluded) { excluded[parsed.excluded] = (excluded[parsed.excluded] || 0) + 1; continue; }
    let resolved = [];
    if (name === 'human' && !ambiguous.has(parsed.object) && accessions.has(parsed.object)) resolved = [accessions.get(parsed.object)];
    if (name === 'yeast') resolved = [...new Set(parsed.orfs.map(orf => `${selected.taxon}:${orf}`).filter(gene => genes.has(gene)))];
    if (resolved.length !== 1) { excluded.unresolved_or_ambiguous = (excluded.unresolved_or_ambiguous || 0) + 1; continue; }
    genes.get(resolved[0]).add(parsed.term);
    admittedRows++;
    evidence[parsed.evidence] = (evidence[parsed.evidence] || 0) + 1;
  }
  const shards = [], allTerms = new Set();
  let rows = [], count = 0;
  async function flush() {
    if (!rows.length) return;
    const raw = encoder.encode(rows.map(x => JSON.stringify(x)).join('\n') + '\n');
    const body = await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer();
    const name = `annotations-${String(shards.length).padStart(5, '0')}.jsonl.gz`, digest = await sha(body);
    await immutable(bucket, prefix + name, body, digest);
    shards.push({name, bytes: body.byteLength, sha256: digest, rows: rows.length});
    count += rows.length; rows = [];
  }
  for (const [gene, terms] of [...genes].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) {
    if (!terms.size) continue;
    for (const term of terms) allTerms.add(term);
    rows.push({gene, terms: [...terms].sort()});
    if (rows.length === 1000) await flush();
  }
  await flush();
  const result = {schema: 'slp.static-go/v1', job, source, identity_manifest_sha256: identityManifestSha,
    identity_key: identityKey, taxon: selected.taxon, input_rows: inputRows, admitted_rows: admittedRows,
    rows: count, roster_genes: genes.size, unique_terms: allTerms.size, ambiguous_accessions: ambiguous.size,
    excluded, evidence, shards, fitted_human_intervention_genes: [],
    filter: {aspects: ['F', 'C'], excluded_evidence: [...excludedEvidence].sort(), exclude_NOT: true,
      date_maximum: '20221231', direct_terms_only: true, names_as_features: false},
    attribution: 'Gene Ontology Consortium, GOA/SGD release 2022-09-19; CC BY 4.0',
    accession_mapping: mappingSource, source_module_sha256: sourceSha};
  if (!result.source_module_sha256 || !count) throw Error(`Empty or unversioned annotation join: genes=${genes.size}, accessions=${accessions.size}, admitted=${admittedRows}`);
  const body = encoder.encode(JSON.stringify(result));
  await immutable(bucket, prefix + 'annotations-manifest.json', body, await sha(body));
  return result;
}
