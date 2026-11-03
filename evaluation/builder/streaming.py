"""SSE chat transport with incremental evidence and normal completion parsing."""
from contextvars import ContextVar
from dataclasses import replace
import json
import time
import httpx
from synthesis.client import ChatClient, _message_text, _redact
from synthesis.types import FatalAPIError, RetryableError

_trace = ContextVar('builder_stream_trace',default=None)

class StreamingChatClient(ChatClient):
    async def complete(self,request,*,trace=None):
        token=_trace.set(trace)
        try:
            request=replace(request,extra={**request.extra,'stream':True,'stream_options':{'include_usage':True}})
            return await super().complete(request,trace=trace)
        finally:_trace.reset(token)

    async def _once(self,body,*,on_response=None):
        content=[];reasoning=[];usage=None;finish=None;model=body.get('model');done=False;reported=False
        metadata={};raw=[];raw_size=0;last_flush=time.monotonic()
        def flush(force=False):
            nonlocal raw_size,last_flush
            if raw and (force or raw_size>=16384 or time.monotonic()-last_flush>=1):
                callback=_trace.get()
                if callback:callback(dict(event='stream_chunk',encoding='utf8-sse-lines',body=_redact(''.join(raw),self._api_key)))
                raw.clear();raw_size=0;last_flush=time.monotonic()
        def envelope(partial=False):
            return dict(**metadata,model=model,assembled_from_stream=True,partial=partial,usage=usage,
                choices=[dict(index=0,finish_reason=finish,message=dict(role='assistant',content=''.join(content),reasoning_content=''.join(reasoning)))])
        def consume(data):
            nonlocal usage,finish,model,done
            if data=='[DONE]':done=True;return
            try:chunk=json.loads(data)
            except ValueError as error:raise RetryableError('Invalid JSON in SSE event') from error
            if 'error' in chunk:raise FatalAPIError(_redact(json.dumps(chunk['error']),self._api_key))
            model=chunk.get('model',model)
            for key in ('id','created','system_fingerprint'):
                if key in chunk:metadata[key]=chunk[key]
            if chunk.get('usage') is not None:usage=chunk['usage']
            for choice in chunk.get('choices',[]):
                if choice.get('index',0)!=0:continue
                delta=choice.get('delta',{})
                if delta.get('tool_calls') or delta.get('function_call'):raise FatalAPIError('Unexpected tool call in evaluation construction')
                for field,parts in [('content',content),('reasoning_content',reasoning)]:
                    value=delta.get(field)
                    if value is not None:
                        if not isinstance(value,str):raise FatalAPIError('Expected textual SSE delta')
                        parts.append(value)
                if choice.get('finish_reason') is not None:finish=choice['finish_reason']
        try:
            async with self._client.stream('POST',self._url,json=body) as response:
                if response.status_code>=400 or 'text/event-stream' not in response.headers.get('content-type',''):
                    await response.aread()
                    if on_response:on_response(response)
                    reported=True
                    if response.status_code in {408,409,429} or response.status_code>=500:raise RetryableError(f'HTTP {response.status_code}: {_redact(response.text[:500],self._api_key)}')
                    if response.status_code>=400:raise FatalAPIError(f'HTTP {response.status_code}: {_redact(response.text[:500],self._api_key)}')
                    return _message_text(response.json())
                data=[]
                async for line in response.aiter_lines():
                    raw.append(line+'\n');raw_size+=len(line)+1;flush()
                    if line.startswith('data:'):
                        data.append(line[5:].removeprefix(' '))
                    elif not line and data:
                        consume('\n'.join(data));data=[]
                    if done:break
                if data:consume('\n'.join(data))
                if not done or finish is None:raise RetryableError('SSE ended without a complete finish/DONE marker')
                payload=envelope()
                if on_response:on_response(httpx.Response(200,json=payload))
                reported=True
                return _message_text(payload)
        except (httpx.TimeoutException,httpx.TransportError) as error:
            raise RetryableError(f'stream failed: {type(error).__name__}: {_redact(str(error),self._api_key)}') from error
        finally:
            flush(True)
            if usage is not None and not reported and on_response:
                on_response(httpx.Response(200,json=envelope(partial=True)))
