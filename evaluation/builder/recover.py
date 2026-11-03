"""One explicit network-recovery attempt, sharing a frozen run's billing caps."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
from dotenv import load_dotenv
from synthesis.client import ChatClient
from synthesis.io import append_jsonl, read_jsonl
from synthesis.types import FatalAPIError, RetryableError
from evaluation.core import canonical_hash, sha256, write_json
from .models import Settings
from .pipeline import Calls, build_job, materialize, now, run_lock, verify

NAME = "manual-recovery-1"

class RecoveryPaused(ValueError):
    pass


def prepare(output, concurrency=8):
    parent = verify(output)
    child = output / NAME
    if not 1 <= concurrency <= parent['config']['concurrency']:
        raise ValueError('Recovery concurrency must not exceed the original run')
    if child.exists():
        plan = json.loads((child/'plan.json').read_text())
        digest = plan.pop('plan_sha256')
        if canonical_hash(plan) != digest or plan['parent_plan_sha256'] != parent['plan_sha256'] or plan['code_sha256'] != sha256(Path(__file__)) or plan['concurrency'] != concurrency:
            raise ValueError('Recovery plan changed')
        for name, key in [('parent-calls.jsonl','parent_calls_sha256'),('parent-outcomes.jsonl','parent_outcomes_sha256')]:
            if sha256(child/name) != plan[key]: raise ValueError('Recovery snapshot changed')
        plan['plan_sha256'] = digest
        return parent, child, plan
    calls = {x['key']:x for x in read_jsonl(output/'calls.jsonl')}
    outcomes = {x['id']:x for x in read_jsonl(output/'outcomes.jsonl')}
    selected = [job['id'] for job in parent['jobs'] if outcomes.get(job['id'],{}).get('error_type') == 'RetryableError']
    failed = [key for key,row in calls.items() if row['state']=='failed' and row.get('error_type')=='RetryableError' and any(key.startswith(identity+'/') for identity in selected)]
    if not selected or not failed: raise ValueError('No network-failed jobs to recover')
    if any('/'+NAME in key for key in calls): raise ValueError('The single recovery attempt was already used')
    spent = sum(x.get('cost_usd',x.get('reservation_usd',0)) for x in calls.values())
    if spent >= parent['config']['max_estimated_usd']: raise ValueError('Original cost allowance is exhausted')
    child.mkdir()
    shutil.copyfile(output/'calls.jsonl',child/'parent-calls.jsonl')
    shutil.copyfile(output/'outcomes.jsonl',child/'parent-outcomes.jsonl')
    maximum = min(parent['config']['max_calls']-len(calls),len(selected)*(2+7*(parent['config']['max_revisions']+1)))
    plan = dict(parent_plan_sha256=parent['plan_sha256'],code_sha256=sha256(Path(__file__)),
        created_at=now(),job_ids=selected,allowed_failed_keys=failed,concurrency=concurrency,
        parent_call_count=len(calls),max_extra_calls=maximum,estimated_peak_spent_including_unknown_reservations=spent,
        parent_calls_sha256=sha256(child/'parent-calls.jsonl'),parent_outcomes_sha256=sha256(child/'parent-outcomes.jsonl'),
        note='One explicit recovery per failed network call; completed calls are reused. Unknown prior usage remains reserved. Stop new requests after another network/API/budget failure.')
    plan['plan_sha256']=canonical_hash(plan)
    write_json(child/'plan.json',plan)
    return parent,child,plan


class RecoveryCalls(Calls):
    def __init__(self,output,parent,client,plan):
        super().__init__(output,parent,client)
        self.recovery_plan=plan
        self.paused=asyncio.Event()
    async def request(self,key,stage,payload,schema):
        previous=self.records.get(key)
        if previous and previous['state']!='complete':
            if key not in self.recovery_plan['allowed_failed_keys']:
                raise ValueError('This failed/uncertain call has no explicit recovery allowance')
            key=key+'/'+NAME
        cached=key in self.records
        if not cached:
            if self.paused.is_set(): raise RecoveryPaused('New requests paused after a network/API/budget failure')
            if len(self.records)-self.recovery_plan['parent_call_count'] >= self.recovery_plan['max_extra_calls']:
                self.paused.set();raise RecoveryPaused('Recovery additional-call limit reached')
        try:
            return await super().request(key,stage,payload,schema)
        except (RetryableError,FatalAPIError):
            self.paused.set();raise
        except ValueError as error:
            if 'token-cost reservation' in str(error) or 'Maximum API call' in str(error):self.paused.set()
            raise


async def execute(output,concurrency=8,client=None):
    parent,child,plan=prepare(output,concurrency)
    if (child/'result.json').exists():return
    settings=Settings.model_validate(parent['config'])
    load_dotenv(Path(parent['root'])/'.env',override=False)
    owned=client is None
    if owned:
        key=os.environ.get(settings.api_key_env)
        if not key:raise ValueError(f'Set {settings.api_key_env}')
        client=ChatClient(base_url=settings.base_url,api_key=key,model=settings.model,timeout=settings.timeout,
            max_retries=0,max_tokens=settings.max_tokens,thinking=settings.thinking,provider=settings.provider)
    try:
        with run_lock(output):
            calls=RecoveryCalls(output,parent,client,plan)
            sources={x['name']:(output/x['snapshot']).read_text() for x in parent['sources']}
            jobs={x['id']:x for x in parent['jobs']}
            semaphore=asyncio.Semaphore(concurrency)
            async def one(identity):
                async with semaphore:
                    if calls.paused.is_set():return
                    try:
                        job=jobs[identity]
                        result=await build_job(job,sources[job['source']],calls)
                    except RecoveryPaused:
                        return
                    except Exception as error:
                        result=dict(id=identity,status='technical_error',error_type=type(error).__name__,error=str(error),versions=[])
                    append_jsonl(output/'outcomes.jsonl',dict(result,recovery_plan_sha256=plan['plan_sha256']))
                    print(json.dumps(dict(id=identity,status=result['status'],time=now())),flush=True)
            await asyncio.gather(*(one(identity) for identity in plan['job_ids']))
            rows=materialize(output,parent)
            write_json(child/'result.json',dict(state='paused' if calls.paused.is_set() else 'finished',
                completed_at=now(),plan_sha256=plan['plan_sha256'],additional_calls=len(calls.records)-plan['parent_call_count'],
                total_calls=len(calls.records),estimated_peak_usd=calls.spent,candidates=len(rows)))
    finally:
        if owned:await client.aclose()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,default=8)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    parent,child,plan=prepare(args.run.resolve(),args.concurrency)
    if args.execute:asyncio.run(execute(args.run.resolve(),args.concurrency))
    else:print(json.dumps(dict(state='preview_only',plan_sha256=plan['plan_sha256'],jobs=len(plan['job_ids']),
        retry_calls=len(plan['allowed_failed_keys']),concurrency=plan['concurrency'],max_extra_calls=plan['max_extra_calls'],
        estimated_peak_spent=plan['estimated_peak_spent_including_unknown_reservations'],max_estimated_usd=parent['config']['max_estimated_usd']),indent=2))

if __name__=='__main__':main()
