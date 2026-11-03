import asyncio,json
import httpx,pytest
from evaluation.builder.streaming import StreamingChatClient
from synthesis.types import ChatRequest,ParseError,RetryableError

class Chunks(httpx.AsyncByteStream):
    def __init__(self,items):self.items=items
    async def __aiter__(self):
        for item in self.items:
            if isinstance(item,Exception):raise item
            yield item

def event(delta=None,finish=None,usage=None):
    return ('data: '+json.dumps(dict(model='deepseek-flash',choices=[dict(index=0,delta=delta or {},finish_reason=finish)],usage=usage))+'\n\n').encode()

def invoke(parts):
    traces=[];bodies=[]
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks(parts))
    async def work():
        c=StreamingChatClient(base_url='https://mock.invalid',api_key='fixture-secret',model='deepseek-flash',thinking=True,provider='deepseek',max_retries=0,transport=httpx.MockTransport(handle))
        try:return await c.complete(ChatRequest(messages=[dict(role='user',content='Return JSON')],max_tokens=131072),trace=traces.append)
        finally:await c.aclose()
    return work,traces,bodies

def test_stream_preserves_thinking_usage_and_wire_settings():
    usage=dict(prompt_tokens=20,completion_tokens=10,prompt_cache_hit_tokens=2,prompt_cache_miss_tokens=18)
    work,traces,bodies=invoke([event({'reasoning_content':'short thought'}),event({'content':'{"ok":'}),event({'content':'true}'}),event(finish='stop',usage=usage),b'data: [DONE]\n\n'])
    assert asyncio.run(work())=='{"ok":true}'
    assert bodies[0]['stream'] and bodies[0]['stream_options']['include_usage'] and bodies[0]['max_tokens']==131072
    response=json.loads(next(x['body'] for x in traces if x['event']=='response'))
    assert response['usage']==usage and response['choices'][0]['message']['reasoning_content']=='short thought'
    assert response['assembled_from_stream']
    assert 'data: [DONE]' in ''.join(x['body'] for x in traces if x['event']=='stream_chunk')

def test_broken_stream_preserves_partial_content_without_claiming_success():
    work,traces,bodies=invoke([event({'content':'partial'}),httpx.RemoteProtocolError('connection lost')])
    with pytest.raises(RetryableError,match='RemoteProtocolError'):asyncio.run(work())
    assert len(bodies)==1
    assert 'partial' in ''.join(x['body'] for x in traces if x['event']=='stream_chunk')
    assert not any(x['event']=='response' for x in traces)

def test_truncated_token_limit_keeps_usage_and_fails():
    work,traces,_=invoke([event({'content':'partial'}),event(finish='length',usage={'prompt_tokens':20,'completion_tokens':131072}),b'data: [DONE]\n\n'])
    with pytest.raises(ParseError,match='token limit'):asyncio.run(work())
    assert json.loads(next(x['body'] for x in traces if x['event']=='response'))['usage']['completion_tokens']==131072

def test_missing_done_is_not_accepted():
    work,traces,_=invoke([event({'content':'{"ok":true}'}),event(finish='stop',usage={'prompt_tokens':20,'completion_tokens':10})])
    with pytest.raises(RetryableError,match='DONE'):asyncio.run(work())
    response=json.loads(next(x['body'] for x in traces if x['event']=='response'))
    assert response['partial'] and response['usage']['completion_tokens']==10

def test_recovery_reuses_successful_alias_and_gates_remaining_phase(tmp_path):
    from evaluation.builder.recover_stream import execute as resume,prepare
    from evaluation.builder.recover import execute as first_recovery
    from test_eval_builder import workspace,execute,client
    from test_eval_recovery import NetworkOnce
    _,output=workspace(tmp_path,revisions=1);api=NetworkOnce();execute(output,api)
    parent,child,plan=prepare(output,1,'toy/section-1')
    async def work():
        c=client(api)
        try:
            with pytest.raises(ValueError,match='completed streaming probe'):
                await resume(output,'remaining',1,'toy/section-1',client=c)
            await resume(output,'probe',1,'toy/section-1',client=c)
            await resume(output,'probe',1,'toy/section-1',client=c)
            await resume(output,'remaining',1,'toy/section-1',client=c)
        finally:await c.aclose()
    asyncio.run(work())
    assert len(api.requests)==10
    assert json.loads((child/'probe-result.json').read_text())['state']=='finished'
    assert json.loads((output/'candidates.jsonl').read_text())['quality_status']=='machine_passed'
