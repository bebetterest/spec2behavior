"""Pause/resume tests: terminal replies survive; successful calls never repeat."""
import asyncio
import json

import httpx
import pytest

from evaluation.builder.models import Case
from evaluation.builder.pipeline import build_job
from evaluation.builder.resume_supplement import ResumableCalls, execute, prepare
from evaluation.builder.supplement import SupplementPaused
from evaluation.core import sha256
from synthesis.io import append_jsonl, read_jsonl
from test_eval_builder import MockAPI, client
from test_eval_supplement import prepared


def snapshot(output, plan):
    for name in ('calls.jsonl', 'outcomes.jsonl'):
        (output / name).touch(exist_ok=True)
    calls = {r['key']:r for r in read_jsonl(output / 'calls.jsonl')}
    outcomes = {r['id']:r for r in read_jsonl(output / 'outcomes.jsonl')}
    file = output / 'resume-jobs.json'
    file.write_text(json.dumps(dict(parent_plan_sha256=plan['plan_sha256'],
        calls_sha256=sha256(output / 'calls.jsonl'), outcomes_sha256=sha256(output / 'outcomes.jsonl'),
        jobs=[dict(id=j['id']) for j in plan['jobs'] if j['id'] not in outcomes or outcomes[j['id']]['status']=='user_paused'])))
    return file


def test_drain_cached_partial_and_explicit_resume_without_rebilling(tmp_path):
    _, output, plan = prepared(tmp_path, cases_per_focus=2)
    stop = output / 'pause.requested'
    class PauseAfterReply(MockAPI):
        def handle(self, request):
            result = super().handle(request)
            stop.write_text('User requested pause')
            return result
    first = PauseAfterReply()
    async def partial():
        c = client(first)
        calls = ResumableCalls(output, plan, c, stop)
        job=plan['jobs'][0]; text=(output / plan['sources'][0]['snapshot']).read_text()
        try:
            with pytest.raises(SupplementPaused):
                await build_job(job, text, calls)
        finally:
            await c.aclose()
        assert len(calls.records)==1
        assert next(iter(calls.records.values()))['state']=='complete'
    asyncio.run(partial())
    append_jsonl(output / 'outcomes.jsonl',dict(id=plan['jobs'][0]['id'],status='user_paused',versions=[]))
    prepare(output, snapshot(output, plan), 'resume-1')
    stop.unlink(); followup=MockAPI()
    async def resume():
        c=client(followup)
        try:
            result=await execute(output,'resume-1',c)
            assert result['state']=='finished' and not result['remaining_ids']
            await execute(output,'resume-1',c)
        finally: await c.aclose()
    asyncio.run(resume())
    assert len(first.requests)==1 and len(followup.requests)==13
    rows=list(read_jsonl(output/'calls.jsonl'))
    assert len(rows)==28 and all(r['state']=='complete' for r in rows[1::2])


def test_stop_file_before_execution_creates_no_paid_attempt(tmp_path):
    _,output,plan=prepared(tmp_path,cases_per_focus=2)
    prepare(output,snapshot(output,plan),'resume-1')
    (output/'pause.requested').touch()
    api=MockAPI()
    async def work():
        c=client(api)
        try: return await execute(output,'resume-1',c)
        finally: await c.aclose()
    result=asyncio.run(work())
    assert result['state']=='user_paused' and result['inflight']==0
    assert len(result['remaining_ids'])==2 and not api.requests
    assert (output/'segments/resume-1/resume-jobs.json').exists()


def test_resume_snapshot_mutation_blocks_before_http(tmp_path):
    _,output,plan=prepared(tmp_path)
    path=snapshot(output,plan)
    append_jsonl(output/'outcomes.jsonl',dict(id='unrelated',status='user_paused'))
    with pytest.raises(ValueError,match='snapshot changed'): prepare(output,path,'resume-1')


def test_failed_request_is_not_resumable(tmp_path):
    _,output,plan=prepared(tmp_path)
    class Fail(MockAPI):
        def handle(self,request): raise httpx.ReadError('fixture disconnected')
    async def fail():
        c=client(Fail());calls=ResumableCalls(output,plan,c,output/'pause.requested')
        try:
            with pytest.raises(Exception):
                await build_job(plan['jobs'][0],(output/plan['sources'][0]['snapshot']).read_text(),calls)
        finally: await c.aclose()
    asyncio.run(fail())
    with pytest.raises(ValueError,match='non-failed jobs'): prepare(output,snapshot(output,plan),'resume-1')


def test_two_inflight_replies_drain_after_stop_without_new_requests(tmp_path):
    _, output, plan = prepared(tmp_path, cases_per_focus=2)
    stop = output / 'pause.requested'
    api = MockAPI()
    async def work():
        both_started = asyncio.Event(); admitted = 0
        async def handle(request):
            nonlocal admitted
            admitted += 1
            if admitted == 2:
                stop.touch(); both_started.set()
            await both_started.wait()
            return api.handle(request)
        from synthesis.client import ChatClient
        c = ChatClient(base_url='https://api.deepseek.com', api_key='fixture-key',
            model='deepseek-flash', thinking=True, max_retries=0, trust_env=False,
            transport=httpx.MockTransport(handle))
        calls = ResumableCalls(output, plan, c, stop)
        try:
            await asyncio.wait_for(asyncio.gather(*(calls.request(j['id']+'/v0/case',
                'generate', dict(REVISION_FEEDBACK=None), Case) for j in plan['jobs'])), timeout=2)
            with pytest.raises(SupplementPaused):
                await calls.request(plan['jobs'][0]['id']+'/v1/case', 'generate',
                    dict(REVISION_FEEDBACK=None), Case)
        finally: await c.aclose()
        assert len(calls.records) == 2 and all(r['state']=='complete' for r in calls.records.values())
    asyncio.run(work())
    assert len(api.requests) == 2
