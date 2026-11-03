"""Explicit streaming recovery: one probe job, then source-balanced remaining jobs."""
import argparse,asyncio,json,os,shutil
from collections import defaultdict,deque
from pathlib import Path
from dotenv import load_dotenv
from evaluation.core import canonical_hash,sha256,write_json
from synthesis.io import append_jsonl,read_jsonl
from synthesis.types import FatalAPIError,RetryableError
from .models import Settings
from .pipeline import Calls,build_job,materialize,now,run_lock,verify
from .streaming import StreamingChatClient

NAME='stream-recovery-1'
class StreamPaused(ValueError):pass

def prepare(output,concurrency=4,probe_id=None):
    parent=verify(output);child=output/NAME
    if parent['config']['provider']!='deepseek':raise ValueError('This recovery is currently verified for DeepSeek')
    if not 1<=concurrency<=parent['config']['concurrency']:raise ValueError('Invalid recovery concurrency')
    if child.exists():
        plan=json.loads((child/'plan.json').read_text());digest=plan.pop('plan_sha256')
        if canonical_hash(plan)!=digest or plan['parent_plan_sha256']!=parent['plan_sha256'] or plan['concurrency']!=concurrency or (probe_id and probe_id!=plan['probe_id']):raise ValueError('Streaming recovery plan changed')
        for name,digest_expected in plan['code'].items():
            if sha256(Path(name))!=digest_expected:raise ValueError('Streaming recovery code changed')
        for name,key in [('parent-calls.jsonl','parent_calls_sha256'),('parent-outcomes.jsonl','parent_outcomes_sha256')]:
            if sha256(child/name)!=plan[key]:raise ValueError('Recovery source snapshot changed')
        plan['plan_sha256']=digest;return parent,child,plan
    calls={x['key']:x for x in read_jsonl(output/'calls.jsonl')};out={x['id']:x for x in read_jsonl(output/'outcomes.jsonl')}
    jobs=[x for x in parent['jobs'] if out.get(x['id'],{}).get('error_type')=='RetryableError']
    if not jobs:raise ValueError('No network-failed jobs remain')
    if any('/'+NAME in key for key in calls):raise ValueError('Streaming recovery was already attempted')
    ids={x['id'] for x in jobs};probe_id=probe_id or jobs[0]['id']
    if probe_id not in ids:raise ValueError('Probe must be a network-failed job')
    groups=defaultdict(list)
    for key,row in calls.items():
        if any(key.startswith(identity+'/') for identity in ids):groups[key.removesuffix('/manual-recovery-1')].append(row)
    routes={}
    for logical,attempts in groups.items():
        completed=[x for x in attempts if x['state']=='complete']
        if completed:routes[logical]=completed[-1]['key']
        elif all(x['state']=='failed' and x.get('error_type')=='RetryableError' for x in attempts):routes[logical]=logical+'/'+NAME
        else:raise ValueError('A selected job also has an unresolved non-network failure')
    by_source={source['name']:deque(x['id'] for x in jobs if x['source']==source['name'] and x['id']!=probe_id) for source in parent['sources']}
    order=[]
    while any(by_source.values()):
        for queue in by_source.values():
            if queue:order.append(queue.popleft())
    spent=sum(x.get('cost_usd',x.get('reservation_usd',0)) for x in calls.values())
    if spent>=parent['config']['max_estimated_usd']:raise ValueError('Original estimate allowance exhausted')
    child.mkdir();shutil.copyfile(output/'calls.jsonl',child/'parent-calls.jsonl');shutil.copyfile(output/'outcomes.jsonl',child/'parent-outcomes.jsonl')
    files=[Path(__file__),Path(__file__).with_name('streaming.py')]
    plan=dict(parent_plan_sha256=parent['plan_sha256'],created_at=now(),concurrency=concurrency,probe_id=probe_id,job_ids=order,
        routes=routes,code={str(p.resolve()):sha256(p) for p in files},parent_call_count=len(calls),
        max_extra_calls=min(parent['config']['max_calls']-len(calls),16*len(jobs)),
        prior_estimated_peak_usd=spent,parent_calls_sha256=sha256(child/'parent-calls.jsonl'),parent_outcomes_sha256=sha256(child/'parent-outcomes.jsonl'),
        transport='SSE stream=true, stream_options.include_usage=true; incremental raw SSE evidence',
        note='Same model/prompts/token cap/global billing caps. One explicit new attempt for unresolved network failures; retain unknown earlier usage reservations; stop new calls after another network/API/budget error.')
    plan['plan_sha256']=canonical_hash(plan);write_json(child/'plan.json',plan)
    return parent,child,plan

class StreamCalls(Calls):
    def __init__(self,output,parent,client,plan):
        super().__init__(output,parent,client);self.recovery_plan=plan;self.paused=asyncio.Event()
    async def request(self,key,stage,payload,schema):
        key=self.recovery_plan['routes'].get(key,key+'/'+NAME)
        if key not in self.records:
            if self.paused.is_set():raise StreamPaused('Stopped after network/API/budget error')
            if len(self.records)-self.recovery_plan['parent_call_count']>=self.recovery_plan['max_extra_calls']:
                self.paused.set();raise StreamPaused('Additional-call limit reached')
        try:return await super().request(key,stage,payload,schema)
        except (RetryableError,FatalAPIError):self.paused.set();raise
        except ValueError as error:
            if 'token-cost reservation' in str(error) or 'Maximum API call' in str(error):self.paused.set()
            raise

async def execute(output,phase,concurrency=4,probe_id=None,client=None):
    parent,child,plan=prepare(output,concurrency,probe_id);result_path=child/(phase+'-result.json')
    if result_path.exists():return
    if phase=='remaining':
        previous=child/'probe-result.json'
        if not previous.exists() or json.loads(previous.read_text())['state']!='finished':raise ValueError('A completed streaming probe is required before the remaining jobs')
    settings=Settings.model_validate(parent['config']);load_dotenv(Path(parent['root'])/'.env',override=False);owned=client is None
    if owned:
        key=os.environ.get(settings.api_key_env)
        if not key:raise ValueError(f'Set {settings.api_key_env}')
        client=StreamingChatClient(base_url=settings.base_url,api_key=key,model=settings.model,timeout=settings.timeout,max_retries=0,max_tokens=settings.max_tokens,thinking=settings.thinking,provider=settings.provider)
    try:
        with run_lock(output):
            calls=StreamCalls(output,parent,client,plan);jobs={x['id']:x for x in parent['jobs']};sources={x['name']:(output/x['snapshot']).read_text() for x in parent['sources']}
            semaphore=asyncio.Semaphore(1 if phase=='probe' else concurrency)
            async def one(identity):
                async with semaphore:
                    if calls.paused.is_set():return
                    try:
                        job=jobs[identity];result=await build_job(job,sources[job['source']],calls)
                    except StreamPaused:return
                    except Exception as error:result=dict(id=identity,status='technical_error',error_type=type(error).__name__,error=str(error),versions=[])
                    append_jsonl(output/'outcomes.jsonl',dict(result,stream_recovery_plan_sha256=plan['plan_sha256']))
                    print(json.dumps(dict(id=identity,status=result['status'],time=now())),flush=True)
            write_json(output/'status.json',dict(state='stream_recovery_'+phase,plan_sha256=plan['plan_sha256'],time=now()))
            await asyncio.gather(*(one(identity) for identity in ([plan['probe_id']] if phase=='probe' else plan['job_ids'])))
            rows=materialize(output,parent)
            result=dict(state='paused' if calls.paused.is_set() else 'finished',phase=phase,completed_at=now(),plan_sha256=plan['plan_sha256'],additional_calls=len(calls.records)-plan['parent_call_count'],total_calls=len(calls.records),estimated_peak_usd=calls.spent,candidates=len(rows))
            write_json(result_path,result);write_json(output/'status.json',dict(result,state='stream_recovery_'+phase+'_'+result['state']))
    finally:
        if owned:await client.aclose()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--concurrency',type=int,default=4);p.add_argument('--probe-id');p.add_argument('--phase',choices=['probe','remaining'],default='probe');p.add_argument('--execute',action='store_true');a=p.parse_args()
    parent,child,plan=prepare(a.run.resolve(),a.concurrency,a.probe_id)
    if a.execute:asyncio.run(execute(a.run.resolve(),a.phase,a.concurrency,a.probe_id))
    else:print(json.dumps(dict(state='preview_only',plan_sha256=plan['plan_sha256'],probe_id=plan['probe_id'],remaining_jobs=len(plan['job_ids']),concurrency=plan['concurrency'],prior_estimated_peak_usd=plan['prior_estimated_peak_usd'],max_estimated_usd=parent['config']['max_estimated_usd']),indent=2))
if __name__=='__main__':main()
