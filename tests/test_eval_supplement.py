"""Supplement contracts: source reuse, fresh cases, blind repair, bounded failure."""
import asyncio
import json

import httpx
import pytest

from evaluation.builder.supplement import execute as supplement, prepare
from evaluation.builder.pipeline import export, verify
from test_eval_builder import MockAPI, PROBES, client, execute, workspace


def parent_run(tmp_path, *, rejected=False):
    root, parent = workspace(tmp_path)
    row = execute(parent, MockAPI())[0]
    decisions = [dict(id=row['id'], candidate_sha256=row['candidate_sha256'],
        decision='reject' if rejected else 'accept', reviewer='Fixture', reviewer_type='codex',
        rationale='Replace the conflicting rubric scale while preserving the focus.',
        probe_labels={p['kind']: p['kind'].endswith('pass') for p in PROBES})]
    (parent / 'decisions.json').write_text(json.dumps(decisions))
    export(parent, parent / 'decisions.json', parent / 'reviewed/combined.jsonl')
    return root, parent


def prepared(tmp_path, *, rejected=False, **options):
    root, parent = parent_run(tmp_path, rejected=rejected)
    config = tmp_path / 'options.json'; config.write_text(json.dumps(dict(concurrency=1, **options)))
    output = tmp_path / 'supplement'
    plan = prepare(parent, output, config)
    return parent, output, plan


def run(output, api, phase):
    async def work():
        c = client(api)
        try: await supplement(output, phase=phase, client=c)
        finally: await c.aclose()
    asyncio.run(work())


def test_source_reuse_fresh_variants_and_no_rebilling(tmp_path):
    parent, output, plan = prepared(tmp_path, cases_per_focus=3)
    assert len(plan['jobs']) == 3
    assert len({j['scenario_domain'] for j in plan['jobs']}) == 3
    before = (parent / 'calls.jsonl').read_bytes()
    api = MockAPI()
    run(output, api, 'probe'); run(output, api, 'remaining'); run(output, api, 'remaining')
    assert len(api.requests) == 21  # Seven fresh case/probe/review/grade calls each; source stages are cached.
    assert (parent / 'calls.jsonl').read_bytes() == before
    rows = [json.loads(line) for line in (output / 'candidates.jsonl').read_text().splitlines()]
    assert len(rows) == 3 and len({row['id'] for row in rows}) == 3
    assert all(row['quality_status'] == 'machine_passed' for row in rows)
    assert all(body['max_tokens'] == 131072 and body['thinking']['type'] == 'enabled' for body in api.requests)


def test_repair_feedback_is_only_visible_to_author(tmp_path):
    _, output, plan = prepared(tmp_path, rejected=True, cases_per_focus=1, repairs=['toy/section-1'])
    assert plan['jobs'][0]['id'].endswith('/repair-01')
    api = MockAPI(); run(output, api, 'probe')
    payloads = []
    for body in api.requests:
        text, schema = body['messages'][-1]['content'].split('\nJSON_SCHEMA:\n')
        payloads.append((json.loads(schema)['title'], json.loads(text.split('\n', 1)[1])))
    author = next(p for title,p in payloads if title == 'Case')
    assert author['REVISION_FEEDBACK']['independent_review']['candidate_sha256']
    for title,p in payloads:
        if title != 'Case': assert 'REVISION_FEEDBACK' not in p
        if title == 'Challenge': assert 'CASE' not in p and 'RUBRIC' not in p
        if title == 'Grade':
            assert not {'PROBES','kind','basis','expected_pass','CONSTRUCTION_REQUIREMENTS','SOURCE'} & set(p)


def test_baseline_and_cache_hashes_are_frozen(tmp_path):
    parent, output, _ = prepared(tmp_path)
    (parent / 'decisions.json').write_text('[]')
    with pytest.raises(ValueError, match='Frozen input'):
        verify(output)


def test_explicit_continuation_skips_failed_call_and_keeps_unknown_reservation(tmp_path):
    _, output, _ = prepared(tmp_path, cases_per_focus=3)
    api = MockAPI(); run(output, api, 'probe')
    class Fails(MockAPI):
        def handle(self, request):
            raise httpx.ReadError('fixture disconnect')
    run(output, Fails(), 'remaining')
    rows = [json.loads(line) for line in (output / 'calls.jsonl').read_text().splitlines()]
    failed = next(row for row in rows if row['state'] == 'failed')
    assert not failed['usage_known'] and failed['cost_usd'] == failed['reservation_usd']
    followup = MockAPI(); run(output, followup, 'continue-1')
    assert len(followup.requests) == 7
    finals = [json.loads(line) for line in (output / 'calls.jsonl').read_text().splitlines()]
    assert sum(row['key'] == failed['key'] for row in finals) == 2
    assert json.loads((output / 'continue-1-result.json').read_text())['unstarted_ids'] == []
    assert (output / 'segments/continue-1/plan.json').exists()


def test_remaining_requires_probe_and_budget_stops_before_call(tmp_path):
    _, output, _ = prepared(tmp_path, max_estimated_usd=0.000001)
    with pytest.raises(ValueError, match='completed three-source probe'):
        run(output, MockAPI(), 'remaining')
    api = MockAPI(); run(output, api, 'probe')
    assert not api.requests
    assert json.loads((output / 'probe-result.json').read_text())['state'] == 'paused'
