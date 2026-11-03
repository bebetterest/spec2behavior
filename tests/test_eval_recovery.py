"""Network recovery preserves evidence, billing caps and the stop condition."""
import asyncio
import json
from pathlib import Path
import httpx
import pytest
from evaluation.builder.recover import RecoveryCalls,RecoveryPaused,execute as recover,prepare
from evaluation.builder.models import Extraction
from evaluation.builder.pipeline import verify
from test_eval_builder import MockAPI,workspace,execute,client

class NetworkOnce(MockAPI):
    def __post_init__(self):
        super().__post_init__();self.network_failed=False
    def handle(self,request):
        body=json.loads(request.content)
        if not self.network_failed and '"CASE":' in body['messages'][-1]['content']:
            self.network_failed=True;self.requests.append(body)
            return httpx.Response(503,text='Temporary network failure')
        return super().handle(request)

def test_recovery_reuses_completed_requests_and_preserves_failed_record(tmp_path):
    _,output=workspace(tmp_path,revisions=1);api=NetworkOnce()
    assert execute(output,api)==[] and len(api.requests)==5
    original=(output/'calls.jsonl').read_text()
    async def work():
        c=client(api)
        try:
            await recover(output,concurrency=1,client=c)
            await recover(output,concurrency=1,client=c)
        finally:await c.aclose()
    asyncio.run(work())
    assert len(api.requests)==10
    assert (output/'calls.jsonl').read_text().startswith(original)
    candidate=json.loads((output/'candidates.jsonl').read_text())
    assert candidate['quality_status']=='machine_passed'
    result=json.loads((output/'manual-recovery-1/result.json').read_text())
    assert result['additional_calls']==5 and result['state']=='finished'

def test_recovery_circuit_breaker_prevents_new_calls(tmp_path):
    _,output=workspace(tmp_path,revisions=1);api=NetworkOnce();execute(output,api)
    parent,_,plan=prepare(output,1)
    def fail(request):return httpx.Response(503,text='Still disconnected')
    async def work():
        from synthesis.client import ChatClient
        from synthesis.types import RetryableError
        c=ChatClient(base_url='https://mock.invalid',api_key='fixture',model='deepseek-flash',thinking=True,
                     provider='deepseek',max_retries=0,transport=httpx.MockTransport(fail))
        calls=RecoveryCalls(output,parent,c,plan)
        try:
            with pytest.raises(RetryableError):
                await calls.request('new-network-request','extract',{},Extraction)
            with pytest.raises(RecoveryPaused):
                await calls.request('must-not-start','extract',{},Extraction)
            assert 'must-not-start' not in calls.records
        finally:await c.aclose()
    asyncio.run(work())

def test_schema_failures_are_not_replayed_as_network_failures(tmp_path):
    _,output=workspace(tmp_path,revisions=1);api=MockAPI(invalid_extract=True);execute(output,api)
    with pytest.raises(ValueError,match='No network-failed'):
        prepare(output,1)
