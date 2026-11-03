"""Explicit, snapshot-bound continuation with a stop file that drains live calls."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil

from dotenv import load_dotenv

from evaluation.core import canonical_hash, sha256, write_json
from synthesis.io import append_jsonl, read_jsonl
from .models import Settings
from .pipeline import build_job, materialize, now, run_lock, verify
from .streaming import StreamingChatClient
from .supplement import SupplementCalls, SupplementPaused


def latest(path, key):
    return {row[key]: row for row in read_jsonl(path)}


def resumable(plan, outcomes, records):
    failed = {key.rsplit('/v', 1)[0] for key, row in records.items() if row['state'] != 'complete'}
    return [job['id'] for job in plan['jobs']
            if (job['id'] not in outcomes or outcomes[job['id']]['status'] == 'user_paused')
            and job['id'] not in failed]


def prepare(output: Path, snapshot: Path, segment: str):
    output = output.resolve(); snapshot = snapshot.resolve()
    plan = verify(output)
    if not re.fullmatch(r'resume-[1-9][0-9]*', segment):
        raise ValueError('Use a numbered resume-N segment')
    if (output / '.lock').exists():
        raise ValueError('Generation remains active')
    saved = json.loads(snapshot.read_text())
    if saved['parent_plan_sha256'] != plan['plan_sha256']:
        raise ValueError('Resume snapshot belongs to another plan')
    for filename in ('calls.jsonl', 'outcomes.jsonl'):
        if sha256(output / filename) != saved[filename.replace('.jsonl', '_sha256')]:
            raise ValueError('Resume ledger snapshot changed')
    records = latest(output / 'calls.jsonl', 'key')
    if any(row['state'] == 'started' for row in records.values()):
        raise ValueError('Unresolved physical request; inspect it before resuming')
    outcomes = latest(output / 'outcomes.jsonl', 'id')
    identities = [job['id'] for job in saved['jobs']]
    expected = resumable(plan, outcomes, records)
    if len(identities) != len(set(identities)) or set(identities) != set(expected):
        raise ValueError('Resume allowlist must cover exactly the unfinished non-failed jobs')
    destination = output / 'segments' / segment
    destination.mkdir(parents=True, exist_ok=False)
    for filename in ('calls.jsonl', 'outcomes.jsonl'):
        shutil.copyfile(output / filename, destination / ('before-' + filename))
    shutil.copyfile(snapshot, destination / 'input-resume-jobs.json')
    value = dict(created_at=now(), parent_plan_sha256=plan['plan_sha256'],
        segment=segment, jobs=identities, stop_file=str(output / 'pause.requested'),
        code={str(Path(__file__).resolve()): sha256(Path(__file__))},
        inputs={str(destination / name): sha256(destination / name)
                for name in ('before-calls.jsonl', 'before-outcomes.jsonl', 'input-resume-jobs.json')},
        note='Explicit user resume. Reuse completed calls; never replay failed calls. Parent settings and total budget remain binding.')
    value['plan_sha256'] = canonical_hash(value)
    write_json(destination / 'plan.json', value)
    return value


def verify_segment(output, segment):
    plan = verify(output)
    folder = output / 'segments' / segment
    child = json.loads((folder / 'plan.json').read_text())
    digest = child.pop('plan_sha256')
    if canonical_hash(child) != digest or child['parent_plan_sha256'] != plan['plan_sha256']:
        raise ValueError('Resume plan changed')
    child['plan_sha256'] = digest
    for filename, expected in {**child['inputs'], **child['code']}.items():
        if sha256(Path(filename)) != expected:
            raise ValueError('Frozen resume input/code changed: ' + filename)
    return plan, child, folder


class ResumableCalls(SupplementCalls):
    """Check stop intent before admitting a new paid request; cached calls stay reusable."""
    def __init__(self, output, plan, client, stop_file):
        super().__init__(output, plan, client)
        self.stop_file = stop_file

    async def request(self, key, stage, payload, schema):
        if stage not in {'extract', 'point_review'} and key not in self.records:
            if self.stop_file.exists():
                self.stopped.set()
            if self.stopped.is_set():
                raise SupplementPaused('Stop requested; preserve partial successful calls')
        return await super().request(key, stage, payload, schema)


async def execute(output: Path, segment: str, client=None):
    output = output.resolve()
    plan, child, folder = verify_segment(output, segment)
    result_path = folder / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    settings = Settings.model_validate(plan['config'])
    owned = client is None
    if owned:
        load_dotenv(Path(plan['root']) / '.env', override=False)
        client = StreamingChatClient(base_url=settings.base_url, api_key=os.environ[settings.api_key_env],
            model=settings.model, timeout=settings.timeout, max_retries=0, max_tokens=settings.max_tokens,
            thinking=settings.thinking, provider=settings.provider, trust_env=False)
    try:
        with run_lock(output):
            for filename in ('calls.jsonl', 'outcomes.jsonl'):
                if sha256(output / filename) != sha256(folder / ('before-' + filename)):
                    raise ValueError('Current ledger differs from the approved resume snapshot')
            stop_file = Path(child['stop_file'])
            calls = ResumableCalls(output, plan, client, stop_file)
            jobs = {job['id']: job for job in plan['jobs']}
            semaphore = asyncio.Semaphore(settings.concurrency)
            sources = {s['name']: (output / s['snapshot']).read_text() for s in plan['sources']}
            async def one(identity):
                async with semaphore:
                    if calls.stopped.is_set() or stop_file.exists():
                        calls.stopped.set(); return
                    job = jobs[identity]
                    try:
                        outcome = await build_job(job, sources[job['source']], calls)
                    except SupplementPaused:
                        return
                    except Exception as error:
                        outcome = dict(id=identity, status='technical_error', error_type=type(error).__name__,
                            error=str(error), versions=[])
                    append_jsonl(output / 'outcomes.jsonl', outcome)
                    print(json.dumps(dict(id=identity, status=outcome['status'], time=now())), flush=True)
            write_json(output / 'status.json', dict(state='resume_running', segment=segment,
                started_at=now(), parent_plan_sha256=plan['plan_sha256'], resume_plan_sha256=child['plan_sha256']))
            await asyncio.gather(*(one(identity) for identity in child['jobs']))
            rows = materialize(output, plan)
            outcomes = latest(output / 'outcomes.jsonl', 'id')
            remaining = resumable(plan, outcomes, calls.records)
            if any(row['state'] == 'started' for row in calls.records.values()):
                raise ValueError('Request remains in flight after workers drained')
            state = 'user_paused' if stop_file.exists() else 'stopped_error' if calls.stopped.is_set() else 'finished'
            result = dict(state=state, segment=segment, completed_at=now(), automatic_resume=False,
                generation_must_remain_paused=state == 'user_paused', inflight=0,
                plan_sha256=plan['plan_sha256'], resume_plan_sha256=child['plan_sha256'],
                planned_units=len(plan['jobs']), completed_units=sum(x['status'] != 'user_paused' for x in outcomes.values()),
                remaining_ids=remaining, candidates=len(rows), physical_api_calls=len(calls.records),
                statuses=dict(Counter(x['status'] for x in outcomes.values())),
                known_peak_usd=sum(x.get('cost_usd', 0) for x in calls.records.values() if x.get('usage_known')),
                unknown_reserved_usd=sum(x.get('cost_usd', 0) for x in calls.records.values() if not x.get('usage_known')),
                unknown_usage_calls=sum(not x.get('usage_known') for x in calls.records.values()))
            if remaining:
                write_json(folder / 'resume-jobs.json', dict(parent_plan_sha256=plan['plan_sha256'],
                    calls_sha256=sha256(output / 'calls.jsonl'), outcomes_sha256=sha256(output / 'outcomes.jsonl'),
                    jobs=[dict(id=identity) for identity in remaining], automatic_resume=False))
            write_json(result_path, result); write_json(output / 'status.json', result)
            write_json(output / 'WORK_STATE.json', result)
            return result
    finally:
        if owned:
            await client.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('prepare'); make.add_argument('--plan', type=Path, required=True)
    make.add_argument('--snapshot', type=Path, required=True); make.add_argument('--segment', required=True)
    run = sub.add_parser('run'); run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--segment', required=True); run.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        value = prepare(args.plan, args.snapshot, args.segment)
        print(json.dumps(dict(state='prepared', jobs=len(value['jobs']), plan_sha256=value['plan_sha256'])))
    elif args.execute:
        result = asyncio.run(execute(args.plan, args.segment))
        print(json.dumps(result, indent=2))
    else:
        plan, child, _ = verify_segment(args.plan.resolve(), args.segment)
        print(json.dumps(dict(state='preview_only', jobs=len(child['jobs']), config=plan['config']), indent=2))


if __name__ == '__main__':
    main()
