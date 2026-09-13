// Short-lived capabilities for exact objects. The reusable token never leaves
// the Mac/Cloudflare control plane. These cannot list, delete or choose new keys.
const encoder = new TextEncoder();
import readinessInputs from './readiness-inputs.json' with {type:'json'};
import mixedReadinessInputs from './mixed-readiness-inputs.json' with {type:'json'};
const to64 = bytes => btoa(String.fromCharCode(...bytes)).replaceAll('+','-').replaceAll('/','_').replaceAll('=','');
const from64 = text => Uint8Array.from(atob(text.replaceAll('-','+').replaceAll('_','/')), c => c.charCodeAt(0));
const json = (body,status=200) => Response.json(body,{status,headers:{'Cache-Control':'no-store'}});
async function signingKey(secret) {
  return crypto.subtle.importKey('raw',encoder.encode(secret),{name:'HMAC',hash:'SHA-256'},false,['sign','verify']);
}

export const isCampaignJob = job => /^campaign-r2-[a-zA-Z0-9_-]{1,150}$/.test(job || '');
export function campaignPermission(p) {
  const bounded = Number.isInteger(p.max_bytes) && p.max_bytes > 0 && p.max_bytes <= 17*1024*1024;
  if(!bounded || !/^[a-zA-Z0-9_-]+\.(json|jsonl\.gz|bin\.gz)$/.test(p.name || '')) return false;
  if(isCampaignJob(p.job)) return ['GET','PUT'].includes(p.method);
  return p.method==='GET' && ['protocol-r2-20260912-v4','feng-folds-r2-20260912-v1'].includes(p.job)
    && /-(train|valid|test)-(manifest\.json|\d{5}\.jsonl\.gz)$/.test(p.name);
}
export async function mintCampaignTickets(secret,origin,permissions,now=Date.now()) {
  if(!Array.isArray(permissions) || !permissions.length || permissions.length>1024 || !permissions.every(campaignPermission)) throw Error('Invalid exact campaign scope');
  return mint(secret,origin,permissions,'exact files for one declared campaign operation',now);
}

export function featurePermissions() {
  const input='identity-r2-20260912-v5', output='esm-r2-20260912-v1';
  const permissions=[{job:input,name:'sequence-inputs-manifest.json',method:'GET',max_bytes:1048576}];
  for(let i=0;i<5;i++) permissions.push({job:input,name:`sequence-inputs-${String(i).padStart(5,'0')}.jsonl.gz`,method:'GET',max_bytes:32*1024*1024});
  for (const [file,count] of [['sequence-npy',16],['annotation-npy',1],['known-npy',1],['genes-json',1],['manifest-json',1]]) {
    for(let i=0;i<count;i++) permissions.push({job:output,name:`${file}-part${String(i).padStart(5,'0')}.bin.gz`,method:'PUT',max_bytes:17*1024*1024});
  }
  for(const name of ['artifact.json','complete.json','failed.json']) permissions.push({job:output,name,method:'PUT',max_bytes:1048576});
  return permissions;
}

export async function mintFeatureTickets(secret,origin,now=Date.now()) {
  return mint(secret,origin,featurePermissions(),'frozen ESM inputs and outputs only',now);
}

export function optimizerPermissions(){
  const job='optimizer-readiness-r2-20260912-v1',permissions=[];
  for(let i=0;i<96;i++) for(const method of ['GET','PUT']) permissions.push({job,name:`checkpoint-pt-part${String(i).padStart(5,'0')}.bin.gz`,method,max_bytes:17*1024*1024});
  for(const name of ['artifact.json','complete.json','failed.json']) for(const method of ['GET','PUT']) permissions.push({job,name,method,max_bytes:4*1024*1024});
  return permissions;
}
export async function mintOptimizerTickets(secret,origin,now=Date.now()){
  return mint(secret,origin,optimizerPermissions(),'one disposable optimizer checkpoint roundtrip only',now);
}

export function readinessPermissions(mixed=false,final=false) {
  const job=final?'readiness-r2-20260912-v3':mixed?'readiness-r2-20260912-v2':'readiness-r2-20260912-v1', permissions=[...(mixed?mixedReadinessInputs:readinessInputs)];
  const parts={'weights-pt':32,'sequence-npy':16,'annotation-npy':1,'known-npy':1,'genes-json':1,
    'features-manifest-json':1,'config-json':1,'vocabulary-json':1,'basal-json':mixed?8:1,'bundle-json':mixed?4:1,
    'model-py':1,'records-py':1,'data-py':1,'artifact-py':1,'inference-py':1,'requirements-linux-cu128-lock':1,...(mixed?{'baselines-py':1}:{})};
  for(const [file,count] of Object.entries(parts)) for(let i=0;i<count;i++) for(const method of ['GET','PUT'])
    permissions.push({job,name:`${file}-part${String(i).padStart(5,'0')}.bin.gz`,method,max_bytes:17*1024*1024});
  for(const name of ['artifact.json','complete.json','failed.json']) for(const method of ['GET','PUT'])
    permissions.push({job,name,method,max_bytes:(mixed?4:1)*1024*1024});
  return permissions;
}

export async function mintReadinessTickets(secret,origin,now=Date.now(),mixed=false,final=false) {
  return mint(secret,origin,readinessPermissions(mixed,final),'exact corpus inputs and disposable readiness artifacts only',now);
}

async function mint(secret,origin,permissions,purpose,now) {
  const key=await signingKey(secret), expires=Math.floor(now/1000)+10800, result=[];
  for(const permission of permissions) {
    const body=encoder.encode(JSON.stringify({...permission,expires}));
    const encoded=to64(body), signature=to64(new Uint8Array(await crypto.subtle.sign('HMAC',key,body)));
    result.push({...permission,expires,url:`${origin}/transfer/${encoded}/${signature}`});
  }
  return {schema:'slp.object-capabilities/v1',purpose,tickets:result};
}

export async function transfer(request,env,now=Date.now()) {
  try {
    const match=new URL(request.url).pathname.match(/^\/transfer\/([A-Za-z0-9_-]{1,1000})\/([A-Za-z0-9_-]{43})$/);
    if(!match) return json({error:'Invalid capability'},403);
    const body=from64(match[1]), valid=await crypto.subtle.verify('HMAC',await signingKey(env.ACCESS_TOKEN),from64(match[2]),body);
    if(!valid) return json({error:'Invalid capability'},403);
    const p=JSON.parse(new TextDecoder().decode(body));
    if(p.expires<Math.floor(now/1000) || request.method!==p.method || !(campaignPermission(p) || [...featurePermissions(),...optimizerPermissions(),...readinessPermissions(),...readinessPermissions(true),...readinessPermissions(true,true)].some(v=>v.job===p.job&&v.name===p.name&&v.method===p.method&&v.max_bytes===p.max_bytes))) return json({error:'Expired or out-of-scope capability'},403);
    const key=isCampaignJob(p.job)?`slp/runs/slp-1.2-r2/${p.job}/${p.name}`:`slp/prepared/${p.job}/${p.name}`;
    if(p.method==='GET') {
      const object=await env.CORPUS.get(key);
      if(!object || object.size>p.max_bytes) return json({error:'Missing or oversized object'},404);
      return new Response(object.body,{headers:{'Cache-Control':'no-store','Content-Length':String(object.size)}});
    }
    const bytes=Number(request.headers.get('Content-Length')), sha256=request.headers.get('X-Content-SHA256');
    if(!Number.isInteger(bytes)||bytes<1||bytes>p.max_bytes||!/^[a-f0-9]{64}$/.test(sha256||'')) return json({error:'Invalid bounded output'},400);
    const old=await env.CORPUS.head(key);
    if(old) return json({state:old.size===bytes&&old.customMetadata.sha256===sha256?'existing':'conflict'},old.size===bytes&&old.customMetadata.sha256===sha256?200:409);
    const stored=await env.CORPUS.put(key,request.body,{onlyIf:new Headers({'If-None-Match':'*'}),sha256,customMetadata:{sha256,job:p.job},httpMetadata:{contentType:'application/octet-stream'}});
    return json({state:stored?'stored':'conflict'},stored?201:409);
  } catch { return json({error:'Invalid capability or transfer'},403); }
}
