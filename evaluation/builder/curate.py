"""Apply explicit independent feedback within a run's remaining revision/billing limits.

Raw API records are never edited. A child plan binds reviewer feedback and this
implementation; the parent call ledger remains append-only and shares its caps.
"""
from __future__ import annotations
import argparse
import asyncio
from contextvars import ContextVar
from dataclasses import replace
import json
from pathlib import Path
import re
import random
import shutil
from typing import Literal

from pydantic import Field
from synthesis.client import ChatClient
from synthesis.io import append_jsonl, read_jsonl
from evaluation.core import canonical_hash, sha256, write_json
from .models import Case, Challenge, Extraction, Grade, Record, Review, Settings, Span, Text
from .citations import resolve_citation_markers
from .pipeline import Calls, build_job, materialize, now, numbered, run_lock, verify, verify_spans

_effort = ContextVar("curation_review_effort", default=None)

class Revision(Record):
    id: Text
    case_sha256: Text
    reviewer: Text
    reviewer_type: Literal["human", "codex"]
    issues: list[Text] = Field(min_length=1)
    evidence: list[Span] = Field(min_length=1)

class Changes(Record):
    resume: list[Text] = Field(default_factory=list)
    revisions: list[Revision] = Field(default_factory=list)




class ReviewClient:
    def __init__(self, client): self.client = client
    async def complete(self, request, *, trace):
        if _effort.get() and self.client.thinking is not False:
            extra = {"reasoning": {"enabled": True, "effort": _effort.get()}} if self.client.provider == "openrouter" else {"reasoning_effort": _effort.get()}
            request = replace(request, extra={**request.extra, **extra})
        return await self.client.complete(request, trace=trace)


class CuratedCalls(Calls):
    def __init__(self, output, plan, client, child, sources, effort):
        super().__init__(output, plan, ReviewClient(client))
        self.child = child; self.sources = sources; self.effort = effort
        child_settings = json.loads((child / "plan.json").read_text())
        self.initial_calls = child_settings["parent_call_count"]
        self.allowed_extra = child_settings["max_extra_calls"]

    async def request(self, key, stage, payload, schema):
        if key not in self.records and len(self.records) - self.initial_calls >= self.allowed_extra:
            raise ValueError("Curation's additional-call allowance is exhausted")
        token = _effort.set(self.effort if stage in {"review", "point_review"} else None)
        try:
            result = await super().request(key, stage, payload, schema)
        finally:
            _effort.reset(token)
        spans = result.focus.evidence if isinstance(result, Extraction) and result.focus else result.evidence if isinstance(result, Review) else []
        source = next(text for identity, text in self.sources.items() if key.startswith(identity + "/"))
        for span in spans:
            repair = resolve_citation_markers(span, source)
            if repair:
                append_jsonl(self.child / "source-normalizations.jsonl", dict(call_key=key, **repair))
        return result


def parent_record(calls, key):
    row = calls.get(key)
    if not row or row["state"] != "complete":
        raise ValueError(f"Missing completed parent call: {key}")
    return row["result"]


def child_plan(output, changes_path, name, effort):
    parent = verify(output)
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError("Use a lowercase-hyphenated curation name")
    changes = Changes.model_validate_json(changes_path.read_text())
    ids = changes.resume + [row.id for row in changes.revisions]
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= {job["id"] for job in parent["jobs"]}:
        raise ValueError("Curation requires distinct existing job IDs")
    if changes.revisions and parent["config"]["max_revisions"] < 1:
        raise ValueError("The parent plan does not allow a content revision")
    folder = output / name
    if folder.exists():
        existing = json.loads((folder / "plan.json").read_text())
        digest = existing.pop("plan_sha256")
        if canonical_hash(existing) != digest or existing["changes_sha256"] != sha256(changes_path) or existing["code_sha256"] != sha256(Path(__file__)) or existing["review_effort"] != effort:
            raise ValueError("Curation plan changed; do not replay it")
        if existing["parent_plan_sha256"] != parent["plan_sha256"]:
            raise ValueError("Parent plan changed")
        for filename, key in (("parent-calls.jsonl", "parent_calls_sha256"), ("parent-outcomes.jsonl", "parent_outcomes_sha256")):
            if sha256(folder / filename) != existing[key]:
                raise ValueError("Curation source snapshot changed")
        existing["plan_sha256"] = digest
        return parent, changes, folder, existing
    calls = {}
    for row in read_jsonl(output / "calls.jsonl"): calls[row["key"]] = row
    jobs = {job["id"]: job for job in parent["jobs"]}
    source_map = {source["name"]: (output / source["snapshot"]).read_text() for source in parent["sources"]}
    for revision in changes.revisions:
        prior_case = parent_record(calls, revision.id + "/v0/case")
        if canonical_hash(prior_case) != revision.case_sha256:
            raise ValueError("Independent feedback names a different case version")
        verify_spans(revision.evidence, source_map[jobs[revision.id]["source"]])
        if any(key.startswith(revision.id + "/v1/") for key in calls):
            raise ValueError("This case has already used its single revision")
    per_job_maximum = 2 + 7 * (parent["config"]["max_revisions"] + 1)
    maximum = 7 * len(changes.revisions) + sum(max(0, per_job_maximum - sum(key.startswith(identity + "/") for key in calls)) for identity in changes.resume)
    if len(calls) + maximum > parent["config"]["max_calls"]:
        raise ValueError("Curation would exceed the original call allowance")
    folder.mkdir()
    shutil.copyfile(output / "calls.jsonl", folder / "parent-calls.jsonl")
    shutil.copyfile(output / "outcomes.jsonl", folder / "parent-outcomes.jsonl")
    shutil.copyfile(changes_path, folder / "changes.json")
    plan = dict(parent_plan_sha256=parent["plan_sha256"], changes_sha256=sha256(changes_path),
        code_sha256=sha256(Path(__file__)), created_at=now(), max_extra_calls=maximum,
        review_effort=effort, parent_call_count=len(calls), parent_calls_sha256=sha256(folder / "parent-calls.jsonl"),
        parent_outcomes_sha256=sha256(folder / "parent-outcomes.jsonl"),
        note="Completed requests are reused. Failed/uncertain requests are not retried; independent feedback creates a new case revision.")
    plan["plan_sha256"] = canonical_hash(plan)
    write_json(folder / "plan.json", plan)
    return parent, changes, folder, plan


async def revise(job, source, feedback, calls):
    prefix = job["id"]
    extraction = Extraction.model_validate(parent_record(calls.records, prefix + "/extract"))
    focus = extraction.focus
    if focus is None: raise ValueError("No source focus to revise")
    point_review = Review.model_validate(parent_record(calls.records, prefix + "/point-review"))
    for span in [*focus.evidence, *point_review.evidence]:
        repair = resolve_citation_markers(span, source)
        if repair: append_jsonl(calls.child / "source-normalizations.jsonl", dict(call_key=prefix, **repair))
    verify_spans(focus.evidence, source, job["section"])
    if point_review.verdict != "accept": raise ValueError("Source focus has not passed review")
    prior = parent_record(calls.records, prefix + "/v0/case")
    context = dict(SOURCE=numbered(source), PRIMARY_SECTION=job["section"])
    case = await calls.request(prefix + "/v1/case", "generate", dict(context,
        FOCUS=focus.model_dump(mode="json"), SCENARIO_DOMAIN=job["scenario_domain"],
        REVISION_FEEDBACK=dict(previous_case=prior, independent_review=feedback.model_dump(mode="json"))), Case)
    challenge = await calls.request(prefix + "/v1/challenge", "challenge", dict(context,
        FOCUS=focus.model_dump(mode="json"), CONVERSATION=[message.model_dump(mode="json") for message in case.messages]), Challenge)
    review = await calls.request(prefix + "/v1/review", "review", dict(context,
        FOCUS=focus.model_dump(mode="json"), CASE=case.model_dump(mode="json"), PROBES=challenge.model_dump(mode="json")), Review)
    verify_spans(review.evidence, source)
    version = dict(revision=1, case=case.model_dump(mode="json"), challenge=challenge.model_dump(mode="json"),
        review=review.model_dump(mode="json"), grades=[], independent_feedback=feedback.model_dump(mode="json"))
    if review.verdict == "accept":
        order = list(challenge.probes)
        random.Random(canonical_hash(dict(case=version["case"], revision=1))).shuffle(order)
        for probe in order:
            payload = dict(SECTION_ID=job["section"]["id"], CONVERSATION=version["case"]["messages"], ANSWER=probe.response, RUBRIC=case.rubric)
            if calls.settings.grading_mode == "full-spec": payload["SOURCE"] = context["SOURCE"]
            grade = await calls.request(prefix + "/v1/grade/" + probe.kind, "grade", payload, Grade)
            if grade.section_id != job["section"]["id"]: raise ValueError("Wrong section_id")
            version["grades"].append(dict(kind=probe.kind, expected_pass=probe.kind.endswith("pass"), observed_pass=grade.compliance >= 6, **grade.model_dump(mode="json")))
    passed = len(version["grades"]) == 4 and all(grade["expected_pass"] == grade["observed_pass"] for grade in version["grades"])
    append_jsonl(calls.output / "versions.jsonl", dict(id=prefix, **version))
    return dict(id=prefix, extraction=extraction.model_dump(mode="json"), point_review=point_review.model_dump(mode="json"),
        versions=[version], status="machine_passed" if passed else "quality_rejected")


async def execute(output, changes_path, name="curation", effort="low", client=None):
    from dotenv import load_dotenv
    import os
    parent, changes, child, plan = child_plan(output, changes_path, name, effort)
    settings = Settings.model_validate(parent["config"])
    load_dotenv(Path(parent["root"]) / ".env", override=False)
    owned = client is None
    if owned:
        key = os.environ.get(settings.api_key_env)
        if not key: raise ValueError(f"Set {settings.api_key_env}")
        client = ChatClient(base_url=settings.base_url, api_key=key, model=settings.model, timeout=settings.timeout,
            max_retries=0, max_tokens=settings.max_tokens, thinking=settings.thinking, provider=settings.provider)
    jobs = {job["id"]: job for job in parent["jobs"]}
    texts = {source["name"]: (output / source["snapshot"]).read_text() for source in parent["sources"]}
    sources = {identity: texts[job["source"]] for identity, job in jobs.items()}
    try:
        with run_lock(output):
            if (child / "complete.json").exists(): return
            calls = CuratedCalls(output, parent, client, child, sources, effort)
            feedback_by_id = {row.id: row for row in changes.revisions}
            semaphore = asyncio.Semaphore(settings.concurrency)
            async def one(identity):
                async with semaphore:
                    try:
                        result = await revise(jobs[identity], sources[identity], feedback_by_id[identity], calls) if identity in feedback_by_id else await build_job(jobs[identity], sources[identity], calls)
                    except Exception as error:
                        result = dict(id=identity, status="technical_error", error_type=type(error).__name__, error=str(error), versions=[])
                    append_jsonl(output / "outcomes.jsonl", dict(result, curation_plan_sha256=plan["plan_sha256"]))
                    print(json.dumps(dict(id=identity, status=result["status"], time=now())), flush=True)
            ids = changes.resume + list(feedback_by_id)
            await asyncio.gather(*(one(identity) for identity in ids))
            rows = materialize(output, parent)
            write_json(child / "complete.json", dict(completed_at=now(), additional_calls=len(calls.records) - calls.initial_calls,
                total_calls=len(calls.records), cost_peak_usd=calls.spent, plan_sha256=plan["plan_sha256"]))
    finally:
        if owned: await client.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--changes", type=Path, required=True)
    parser.add_argument("--name", default="curation")
    parser.add_argument("--review-effort", choices=["low", "high", "max"], default="low")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    parent, changes, child, plan = child_plan(args.run.resolve(), args.changes.resolve(), args.name, args.review_effort)
    if args.execute: asyncio.run(execute(args.run.resolve(), args.changes.resolve(), args.name, args.review_effort))
    else: print(json.dumps(dict(state="preview_only", **plan), indent=2))

if __name__ == "__main__": main()
