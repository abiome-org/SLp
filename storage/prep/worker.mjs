import {Container} from '@cloudflare/containers';
import manifest from '../source-objects.json' with {type: 'json'};
import {JOB, JOBS, PREFIX, MAX_BYTES, ARTIFACT_JOBS, PREPARATION_JOBS, outputKey, json} from './routing.mjs';
import build from './build-receipt.json' with {type: 'json'};
import {mintFeatureTickets, mintReadinessTickets, mintOptimizerTickets, mintCampaignTickets, transfer} from './transfers.mjs';
import {prepareGo} from './prepare_go.mjs';
import goBuild from './go-build-receipt.json' with {type: 'json'};

export class CorpusPrep extends Container {
  defaultPort = 8080;
  // The process also exits on completion or after 30 minutes, independently.
  sleepAfter = '35m';
  envVars = {
    SLP_STORAGE_URL: 'https://slp-corpus-ingest.potteryrage.workers.dev',
    SLP_PREP_URL: 'https://slp-corpus-prep.potteryrage.workers.dev',
    SLP_STORAGE_TOKEN: this.env.ACCESS_TOKEN,
    SLP_JOB: JOB,
  };
}

export default {
  async fetch(request, env) {
    const path = new URL(request.url).pathname;
    if(path.startsWith('/transfer/')) return transfer(request,env);
    if (!env.ACCESS_TOKEN || request.headers.get('Authorization') !== `Bearer ${env.ACCESS_TOKEN}`) {
      return json({error: 'Unauthorized'}, 401);
    }
    if(request.method === 'POST' && path === '/tickets/esm-r2-20260912-v1') return json(await mintFeatureTickets(env.ACCESS_TOKEN,new URL(request.url).origin));
    if(request.method === 'POST' && path === '/tickets/readiness-r2-20260912-v1') return json(await mintReadinessTickets(env.ACCESS_TOKEN,new URL(request.url).origin));
    if(request.method === 'POST' && path === '/tickets/readiness-r2-20260912-v2') return json(await mintReadinessTickets(env.ACCESS_TOKEN,new URL(request.url).origin,Date.now(),true));
    if(request.method === 'POST' && path === '/tickets/readiness-r2-20260912-v3') return json(await mintReadinessTickets(env.ACCESS_TOKEN,new URL(request.url).origin,Date.now(),true,true));
    if(request.method === 'POST' && path === '/tickets/optimizer-readiness-r2-20260912-v1') return json(await mintOptimizerTickets(env.ACCESS_TOKEN,new URL(request.url).origin));
    if(request.method === 'POST' && path === '/tickets/campaign') {
      if(Number(request.headers.get('Content-Length'))>262144) return json({error:'Scope too large'},400);
      try { const body=await request.json(); return json(await mintCampaignTickets(env.ACCESS_TOKEN,new URL(request.url).origin,body.permissions)); }
      catch { return json({error:'Invalid campaign file scope'},400); }
    }
    if (request.method === 'GET' && path === '/manifest') return json(manifest);
    const staticGo = path.match(/^\/static-go\/(human|yeast)$/);
    if (request.method === 'POST' && staticGo) {
      try { return json(await prepareGo(env.CORPUS, staticGo[1], manifest, goBuild.sha256)); }
      catch (error) { return json({error: 'Static GO preparation failed', detail: error.message}, 500); }
    }
    if (request.method === 'GET' && (path === '/status' || path.startsWith('/status/'))) {
      const job=path==='/status'?JOB:path.slice('/status/'.length);
      if(!JOBS.has(job)&&!PREPARATION_JOBS.has(job)&&!ARTIFACT_JOBS.has(job)) return json({error:'Unknown job'},404);
      const prefix=`slp/prepared/${job}/`;
      const objects = [];
      let cursor;
      do {
        const page = await env.CORPUS.list({prefix, limit: 1000, cursor});
        objects.push(...page.objects.map(o => ({key: o.key, bytes: o.size, etag: o.etag})));
        cursor = page.truncated ? page.cursor : undefined;
      } while (cursor);
      return json({job, objects});
    }
    const key = outputKey(path);
    if (key && request.method === 'PUT') {
      if (!key.startsWith(PREFIX) && !ARTIFACT_JOBS.has(key.split('/')[2]) && !PREPARATION_JOBS.has(key.split('/')[2])) return json({error: 'Historical preparation versions are read-only'}, 403);
      const bytes = Number(request.headers.get('Content-Length'));
      const sha256 = request.headers.get('X-Content-SHA256');
      if (!Number.isInteger(bytes) || bytes < 1 || bytes > MAX_BYTES || !/^[a-f0-9]{64}$/.test(sha256 || '')) {
        return json({error: 'Bounded length and SHA256 required'}, 400);
      }
      const old = await env.CORPUS.head(key);
      if (old) return json({state: old.customMetadata.sha256 === sha256 && old.size === bytes ? 'existing' : 'conflict'},
        old.customMetadata.sha256 === sha256 && old.size === bytes ? 200 : 409);
      const stored = await env.CORPUS.put(key, request.body, {
        onlyIf: new Headers({'If-None-Match': '*'}), sha256,
        customMetadata: {sha256, job: key.split('/')[2]}, httpMetadata: {contentType: 'application/octet-stream'},
      });
      return json({state: stored ? 'stored' : 'conflict'}, stored ? 201 : 409);
    }
    if (key && request.method === 'GET') {
      const object = await env.CORPUS.get(key);
      return object ? new Response(object.body, {headers: {'Content-Type': 'application/json', 'Cache-Control': 'no-store'}}) : json({error: 'Missing'}, 404);
    }
    if ((path === '/run' || path.startsWith('/run/')) && request.method === 'POST') {
      const job=path==='/run'?JOB:path.slice('/run/'.length);
      if(job!==JOB&&!PREPARATION_JOBS.has(job)) return json({error:'Unknown executable job'},404);
      const prefix=`slp/prepared/${job}/`;
      if(await env.CORPUS.head(prefix+'claim.json')) return json({error:'Job already claimed; inspect receipts before retrying'},409);
      const container = env.PREP.getByName(job);
      const healthResponse = await container.fetch(new Request('http://container/health'));
      let health;
      try { health = await healthResponse.json(); } catch { health = {}; }
      if (health.build_sha256 !== build.sha256) {
        // This named job has not been claimed. Retire its stale idle image so
        // PID-1 signal handling cannot leave it serving health checks for 15m.
        if(!(await env.CORPUS.head(prefix+'claim.json'))) await container.destroy();
        return json({error: 'Container image rollout is not ready; no job was claimed', expected_build: build.sha256, observed_build: health.build_sha256 || null}, 503);
      }
      // A failed/ambiguous attempt requires explicit inspection and a new job version.
      const claim = await env.CORPUS.put(prefix + 'claim.json', JSON.stringify({job, build_sha256: build.sha256, started_at: new Date().toISOString()}),
        {onlyIf: new Headers({'If-None-Match': '*'})});
      if (!claim) return json({error: 'Job already claimed; inspect receipts before retrying'}, 409);
      return container.fetch(new Request('http://container/run', {method: 'POST', headers: {'X-SLP-Build': build.sha256, 'X-SLP-Job':job}}));
    }
    return json({error: 'Unknown route'}, 404);
  },
};
