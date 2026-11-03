"""Frozen inputs, bounded API construction, source checks and resumable evidence."""
from __future__ import annotations
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
from uuid import uuid4

from synthesis.client import ChatClient
from synthesis.io import append_jsonl, read_jsonl
from synthesis.markdown_sections import _headings_outside_fences
from synthesis.types import ChatRequest
from evaluation.core import canonical_hash, sha256, write_json
from .models import Case, Challenge, Decision, Extraction, Focus, Grade, Review, Settings
from .prompts import COMMON, DOMAINS, STAGES
from .citations import resolve_citation_markers

PACKAGE = Path(__file__).parent
ROOT = PACKAGE.parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve(root, name):
    return (root / name).resolve()


def code_files(_root):
    root = ROOT  # Code lives with the package, independently of the input workspace.
    paths = [*Path(__file__).parent.glob("*.py"),
             *(root / "synthesis" / name for name in ("client.py", "types.py", "io.py", "markdown_sections.py")),
             *(root / "evaluation" / name for name in ("core.py", "audit.py"))]
    return {str(path.resolve()): sha256(path) for path in sorted(paths)}


def sections(text):
    """Use each heading's direct body, with original lines and no subtree overlap."""
    lines = text.splitlines()
    headings = _headings_outside_fences(text)
    result = []
    if not text.strip():
        return []
    if not headings:
        return [dict(id="document", title="Document", line_start=1, line_end=len(lines))]
    if "\n".join(lines[:headings[0][0]]).strip():
        result.append(dict(id="preamble", title="Preamble", line_start=1, line_end=headings[0][0]))
    for index, (start, level, title, anchor) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        if not "\n".join(lines[start + 1:end]).strip():
            continue
        result.append(dict(id=anchor or f"section-{start + 1}", title=title, line_start=start + 1, line_end=end))
    return result


def verify_spans(spans, text, primary=None):
    lines = text.splitlines()
    if not spans:
        raise ValueError("source evidence is required")
    for span in spans:
        if span.line_end < span.line_start or span.line_end > len(lines):
            raise ValueError("source line range is out of bounds")
        if span.quote not in "\n".join(lines[span.line_start - 1:span.line_end]):
            raise ValueError("source quote does not match the declared original lines")
    if primary and not any(primary["line_start"] <= span.line_start <= span.line_end <= primary["line_end"] for span in spans):
        raise ValueError("focus needs evidence in the selected section")


def prepare(root: Path, config: Path, output: Path):
    if output.exists():
        raise ValueError("Output already exists; inspect or resume its frozen plan")
    settings = Settings.model_validate_json(config.read_text())
    inputs = {}; jobs = []; sources = []
    for source in settings.sources:
        path = resolve(root, source.path)
        text = path.read_text()
        selected = [section for section in sections(text) if not source.sections or any(re.search(pattern, section["title"]) for pattern in source.sections)]
        selected = selected[:settings.max_sections_per_source]
        if not selected:
            raise ValueError(f"No matching sections in {source.path}")
        inputs[str(path)] = sha256(path)
        snapshot = f"sources/{len(sources):02d}.md"
        sources.append(dict(name=source.name, path=source.path, snapshot=snapshot, sha256=sha256(path)))
        for section in selected:
            identity = f"{source.name}/{section['id']}"
            jobs.append(dict(id=identity, source=source.name, section=section,
                             scenario_domain=DOMAINS[len(jobs) % len(DOMAINS)]))
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate source/section identity")
    training_rows = 0
    for name in settings.training_data:
        path = resolve(root, name)
        inputs[str(path)] = sha256(path)
        for row in read_jsonl(path):
            if not isinstance(row.get("messages"), list):
                raise ValueError(f"Training row lacks messages: {name}")
            training_rows += 1
    maximum_calls = len(jobs) * (2 + (settings.max_revisions + 1) * 7)
    if settings.max_calls < maximum_calls:
        raise ValueError(f"max_calls must cover the planned worst case ({maximum_calls})")
    output.mkdir(parents=True)
    for source in sources:
        path = output / source["snapshot"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolve(root, source["path"]), path)
    plan = dict(schema_version=1, root=str(root.resolve()), created_at=now(),
        config=settings.model_dump(mode="json"), sources=sources, jobs=jobs,
        inputs=inputs, code=code_files(root), python=sys.version.split()[0],
        packages={name: importlib.metadata.version(name) for name in ("httpx", "pydantic", "python-dotenv")},
        max_calls=maximum_calls, training_rows_screened=training_rows,
        reference="https://alignment.openai.com/model-spec-evals/",
        review_status="Model checks and Codex review do not equal independent human labels.")
    plan["plan_sha256"] = canonical_hash(plan)
    write_json(output / "plan.json", plan)
    write_json(output / "status.json", dict(state="prepared", jobs=len(jobs), max_calls=maximum_calls))
    return plan


def verify(output):
    plan = json.loads((output / "plan.json").read_text())
    digest = plan.pop("plan_sha256")
    if canonical_hash(plan) != digest:
        raise ValueError("Frozen plan changed; use a fresh output directory")
    plan["plan_sha256"] = digest
    for path, expected in {**plan["inputs"], **plan["code"]}.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Frozen input/code changed: {path}")
    for source in plan["sources"]:
        if sha256(output / source["snapshot"]) != source["sha256"]:
            raise ValueError("Frozen source snapshot changed")
    for name, version in plan["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Dependency changed: {name}")
    if sys.version.split()[0] != plan["python"]:
        raise ValueError("Python version changed")
    return plan


@contextmanager
def run_lock(output):
    path = output / ".lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError("Run is locked; inspect the saved PID before removing a stale lock") from None
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def cost(usage, settings, fallback):
    if not usage or "prompt_tokens" not in usage or "completion_tokens" not in usage:
        return fallback, False
    cached = usage.get("prompt_cache_hit_tokens", 0)
    uncached = usage.get("prompt_cache_miss_tokens", max(0, usage["prompt_tokens"] - cached))
    amount = (uncached * settings.input_miss_usd_per_million + cached * settings.input_hit_usd_per_million
              + usage["completion_tokens"] * settings.output_usd_per_million) / 1e6
    return amount, True


class Calls:
    def __init__(self, output, plan, client):
        self.output = output; self.plan = plan; self.client = client
        self.settings = Settings.model_validate(plan["config"])
        self.records = {}
        for row in read_jsonl(output / "calls.jsonl"):
            self.records[row["key"]] = row
        self.spent = sum(row.get("cost_usd", row.get("reservation_usd", 0)) for row in self.records.values())
        self.reserved = 0.0
        self.lock = asyncio.Lock()
        texts = {source["name"]: (output / source["snapshot"]).read_text() for source in plan["sources"]}
        self.sources = {job["id"]: texts[job["source"]] for job in plan["jobs"]}
        self.normalized = {row["normalization_sha256"] for row in read_jsonl(output / "source-normalizations.jsonl")}

    def normalize_evidence(self, key, result):
        spans = result.focus.evidence if isinstance(result, Extraction) and result.focus else result.evidence if isinstance(result, Review) else []
        if not spans:
            return result
        source = next((text for identity, text in self.sources.items() if key.startswith(identity + "/")), None)
        if source is None:
            return result
        for span in spans:
            repair = resolve_citation_markers(span, source)
            if repair:
                record = dict(call_key=key, **repair)
                fingerprint = canonical_hash(record)
                if fingerprint not in self.normalized:
                    append_jsonl(self.output / "source-normalizations.jsonl", dict(record, normalization_sha256=fingerprint))
                    self.normalized.add(fingerprint)
        return result

    async def request(self, key, stage, payload, schema):
        role = "judge" if stage == "grade" else "author" if stage in {"extract", "generate"} else "reviewer"
        model = self.settings.stage_models.get(role, self.settings.model)
        signature = canonical_hash(dict(stage=stage, payload=payload, schema=schema.model_json_schema(), model=model))
        if key in self.records:
            previous = self.records[key]
            if previous["signature"] != signature:
                raise ValueError("Saved call input differs")
            if previous["state"] != "complete":
                raise ValueError(f"Call {key} failed or is uncertain; inspect it rather than replaying")
            return self.normalize_evidence(key, schema.model_validate(previous["result"]))
        seed = uuid4().hex
        messages = [dict(role="system", content=COMMON + STAGES[stage]),
                    dict(role="user", content=f"Random seed: {seed}\n" + json.dumps(payload, ensure_ascii=False)
                         + "\nJSON_SCHEMA:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False))]
        upper_input_tokens = len(json.dumps(messages, ensure_ascii=False).encode()) + 4096
        reservation = (upper_input_tokens * self.settings.input_miss_usd_per_million
                       + self.settings.max_tokens * self.settings.output_usd_per_million) / 1e6
        async with self.lock:
            if len(self.records) >= self.settings.max_calls:
                raise ValueError("Maximum API call count reached")
            if self.spent + self.reserved + reservation > self.settings.max_estimated_usd:
                raise ValueError("Conservative token-cost reservation exceeds the configured run allowance")
            self.reserved += reservation
            start = dict(key=key, stage=stage, signature=signature, state="started", time=now(),
                         generation_seed=seed, model=model, reservation_usd=reservation)
            self.records[key] = start
            append_jsonl(self.output / "calls.jsonl", start)
        usage = None
        def trace(event):
            nonlocal usage
            append_jsonl(self.output / "api.jsonl", dict(job_call=key, **event))
            if event["event"] == "response":
                try:
                    usage = json.loads(event["body"]).get("usage")
                except (ValueError, AttributeError):
                    pass
        try:
            # DeepSeek ignores temperature in thinking mode; do not imply it controls sampling.
            temperature = None if self.settings.provider == "deepseek" and self.settings.thinking else self.settings.temperature
            reply = await self.client.complete(ChatRequest(messages=messages, model=model,
                temperature=temperature, max_tokens=self.settings.max_tokens,
                extra={"response_format": {"type": "json_object"}}, generation_seed=seed), trace=trace)
            result = schema.model_validate_json(reply)
            terminal = dict(start, state="complete", result=result.model_dump(mode="json"))
        except BaseException as error:
            terminal = dict(start, state="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            amount, known = cost(usage, self.settings, reservation)
            terminal.update(finished_at=now(), usage=usage, cost_usd=amount, usage_known=known)
            append_jsonl(self.output / "calls.jsonl", terminal)
            async with self.lock:
                self.records[key] = terminal
                self.reserved -= reservation
                self.spent += amount
        return self.normalize_evidence(key, result)


def numbered(text):
    return "\n".join(f"{index}: {line}" for index, line in enumerate(text.splitlines(), 1))


async def build_job(job, source_text, calls):
    context = dict(SOURCE=numbered(source_text), PRIMARY_SECTION=job["section"])
    prefix = job["id"]
    extraction = await calls.request(prefix + "/extract", "extract", context, Extraction)
    if extraction.focus is None:
        return dict(id=prefix, status="omitted", extraction=extraction.model_dump(mode="json"), versions=[])
    focus = extraction.focus
    verify_spans(focus.evidence, source_text, job["section"])
    if focus.identity_dependent:
        return dict(id=prefix, status="identity_dependent", extraction=extraction.model_dump(mode="json"), versions=[])
    check = await calls.request(prefix + "/point-review", "point_review", dict(context, FOCUS=focus.model_dump(mode="json")), Review)
    verify_spans(check.evidence, source_text)
    result = dict(id=prefix, extraction=extraction.model_dump(mode="json"), point_review=check.model_dump(mode="json"), versions=[])
    if check.verdict != "accept":
        return dict(result, status="focus_needs_review")
    feedback = None
    for revision in range(calls.settings.max_revisions + 1):
        version = dict(revision=revision)
        case = await calls.request(f"{prefix}/v{revision}/case", "generate", dict(context,
            FOCUS=focus.model_dump(mode="json"), SCENARIO_DOMAIN=job["scenario_domain"], REVISION_FEEDBACK=feedback), Case)
        version["case"] = case.model_dump(mode="json")
        # No rubric or author-generated probes are shown to the challenge writer.
        challenge = await calls.request(f"{prefix}/v{revision}/challenge", "challenge", dict(context,
            FOCUS=focus.model_dump(mode="json"), CONVERSATION=[m.model_dump(mode="json") for m in case.messages]), Challenge)
        version["challenge"] = challenge.model_dump(mode="json")
        review = await calls.request(f"{prefix}/v{revision}/review", "review", dict(context,
            FOCUS=focus.model_dump(mode="json"), CASE=case.model_dump(mode="json"), PROBES=challenge.model_dump(mode="json")), Review)
        verify_spans(review.evidence, source_text)
        version["review"] = review.model_dump(mode="json")
        version["grades"] = []
        if review.verdict == "accept":
            order = list(challenge.probes)
            random.Random(canonical_hash(dict(case=version["case"], revision=revision))).shuffle(order)
            for probe in order:
                payload = dict(SECTION_ID=job["section"]["id"], CONVERSATION=version["case"]["messages"],
                               ANSWER=probe.response, RUBRIC=case.rubric)
                if calls.settings.grading_mode == "full-spec":
                    payload["SOURCE"] = context["SOURCE"]
                # Neither the probe kind/label nor its source reasoning is sent to the judge.
                grade = await calls.request(f"{prefix}/v{revision}/grade/{probe.kind}", "grade", payload, Grade)
                if grade.section_id != job["section"]["id"]:
                    raise ValueError("Judge returned a different section_id")
                expected = probe.kind.endswith("pass")
                version["grades"].append(dict(kind=probe.kind, expected_pass=expected,
                    observed_pass=grade.compliance >= 6, **grade.model_dump(mode="json")))
        passed = review.verdict == "accept" and len(version["grades"]) == 4 and all(g["expected_pass"] == g["observed_pass"] for g in version["grades"])
        result["versions"].append(version)
        append_jsonl(calls.output / "versions.jsonl", dict(id=prefix, **version))
        if passed:
            return dict(result, status="machine_passed")
        if review.verdict == "reject":
            break
        feedback = dict(previous_case=version["case"], independent_review=version["review"], calibration=version["grades"])
    return dict(result, status="quality_rejected")


def screen_training(plan, candidates):
    from evaluation.audit import screen_overlap
    training = [row for name in plan["config"]["training_data"] for row in read_jsonl(resolve(Path(plan["root"]), name))]
    records = {"model_spec": [dict(id=row["id"], input=row["case"]["messages"], target=row["case"]["rubric"], metadata={}) for row in candidates]}
    findings = screen_overlap(records, training, {})
    flagged = set()
    for name in ("exact_input_messages", "contained_input_messages_min_4_words", "generated_text_12word_matches", "full_rubric_matches"):
        flagged.update(hit["eval_id"] for hit in findings[name])
    flagged.update(hit["eval_id"] for hit in findings["lexical_neighbours"] if hit["token_sequence_ratio"] >= .8)
    return findings, flagged


def materialize(output, plan):
    outcomes = {row["id"]: row for row in read_jsonl(output / "outcomes.jsonl")}
    candidates = []
    sources = {source["name"]: source for source in plan["sources"]}
    for job in plan["jobs"]:
        result = outcomes.get(job["id"])
        if not result or not result.get("versions"):
            continue
        version = result["versions"][-1]
        source = sources[job["source"]]
        candidate = dict(id=job["id"], collection=job["source"], source=source["path"],
            source_sha256=source["sha256"], section=job["section"],
            focus=result["extraction"]["focus"], case=version["case"], challenge=version["challenge"],
            review=version["review"], grades=version["grades"], revision=version["revision"],
            quality_status=result["status"], plan_sha256=plan["plan_sha256"])
        candidate["candidate_sha256"] = canonical_hash(candidate)
        candidates.append(candidate)
    findings, flagged = screen_training(plan, candidates) if candidates else ({}, set())
    write_json(output / "overlap.json", dict(findings=findings, flagged_ids=sorted(flagged),
        training_rows=plan["training_rows_screened"], limitation="Lexical screening is not proof of semantic independence."))
    for candidate in candidates:
        candidate["overlap_flagged"] = candidate["id"] in flagged
    (output / "candidates.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates))
    records = {}
    for row in read_jsonl(output / "calls.jsonl"):
        records[row["key"]] = row
    totals = dict(calls=len(records), completed_calls=sum(row["state"] == "complete" for row in records.values()),
        failed_or_uncertain_calls=sum(row["state"] != "complete" for row in records.values()),
        estimated_peak_rate_usd=sum(row.get("cost_usd", row.get("reservation_usd", 0)) for row in records.values()),
        unknown_usage_calls=sum(not row.get("usage_known", False) for row in records.values()))
    write_json(output / "summary.json", dict(plan_sha256=plan["plan_sha256"], jobs=len(plan["jobs"]),
        candidates=len(candidates), statuses={state: sum(row["status"] == state for row in outcomes.values()) for state in sorted({row["status"] for row in outcomes.values()})},
        overlap_flagged=len(flagged), usage=totals, source=plan["reference"]))
    lines = ["# Evaluation candidate review", "", "All decisions are pending. Machine checks and Codex review are not human certification.", "",
        "| Candidate | Status | Revision | Lexical flag |", "|---|---|---:|---|"]
    cards = output / "cards"; cards.mkdir(exist_ok=True)
    for index, row in enumerate(candidates, 1):
        filename = f"{index:02d}.md"
        lines.append(f"| [{row['id']}](cards/{filename}) | {row['quality_status']} | {row['revision']} | {row['overlap_flagged']} |")
        card = [f"# {row['case']['title']}", "", f"Candidate `{row['id']}` / `{row['candidate_sha256']}`", "",
            f"Source: `{row['source']}`. Section: {row['section']['title']}", "", "## Focus", "", json.dumps(row["focus"], ensure_ascii=False, indent=2), "", "## Conversation", ""]
        for message in row["case"]["messages"]:
            card += [f"### {message['role']}", "", message["content"], ""]
        card += ["## Rubric", "", row["case"]["rubric"], "", "## Independent probes", ""]
        grades = {grade["kind"]: grade for grade in row["grades"]}
        for probe in row["challenge"]["probes"]:
            card += [f"### {probe['kind']}", "", probe["response"], "", "Source basis: " + probe["basis"], "", "Grade: " + json.dumps(grades.get(probe["kind"]), ensure_ascii=False), ""]
        card += ["## Model review", "", json.dumps(row["review"], ensure_ascii=False, indent=2), ""]
        (cards / filename).write_text("\n".join(card))
    lines += ["", f"Cost estimate at frozen peak rates: ${totals['estimated_peak_rate_usd']:.6f}; unknown-use calls are charged their conservative reservation. Not an invoice.",
        "", "Full source snapshots, requests, replies, reasoning, failed attempts and every revision remain in this local run."]
    (output / "REVIEW.md").write_text("\n".join(lines) + "\n")
    return candidates


async def run(output, client=None):
    from dotenv import load_dotenv
    plan = verify(output)
    settings = Settings.model_validate(plan["config"])
    load_dotenv(Path(plan["root"]) / ".env", override=False)
    owned = client is None
    if owned:
        key = os.environ.get(settings.api_key_env)
        if not key:
            raise ValueError(f"Set {settings.api_key_env} in the local .env or environment")
        client = ChatClient(base_url=settings.base_url, api_key=key, model=settings.model,
            timeout=settings.timeout, max_retries=0, max_tokens=settings.max_tokens,
            thinking=settings.thinking, provider=settings.provider)
    try:
        with run_lock(output):
            write_json(output / "status.json", dict(state="running", started_at=now()))
            calls = Calls(output, plan, client)
            previous = {row["id"] for row in read_jsonl(output / "outcomes.jsonl")}
            sources = {source["name"]: (output / source["snapshot"]).read_text() for source in plan["sources"]}
            semaphore = asyncio.Semaphore(settings.concurrency)
            async def one(job):
                if job["id"] in previous:
                    return
                async with semaphore:
                    try:
                        result = await build_job(job, sources[job["source"]], calls)
                    except Exception as error:
                        result = dict(id=job["id"], status="technical_error", error_type=type(error).__name__, error=str(error), versions=[])
                    append_jsonl(output / "outcomes.jsonl", result)
                    print(json.dumps(dict(id=job["id"], status=result["status"], time=now())), flush=True)
            await asyncio.gather(*(one(job) for job in plan["jobs"]))
            candidates = materialize(output, plan)
            write_json(output / "status.json", dict(state="needs_review", completed_at=now(), candidates=len(candidates)))
            return candidates
    except BaseException as error:
        write_json(output / "status.json", dict(state="interrupted_or_failed", error_type=type(error).__name__, time=now()))
        raise
    finally:
        if owned:
            await client.aclose()


def export(output, decisions_path, destination):
    plan = verify(output)
    if destination.exists():
        raise ValueError("Export destination exists; use a new file to preserve prior data")
    candidates = {row["id"]: row for row in read_jsonl(output / "candidates.jsonl")}
    decisions = [Decision.model_validate(row) for row in json.loads(decisions_path.read_text())]
    if len({row.id for row in decisions}) != len(decisions):
        raise ValueError("Duplicate review decisions")
    kept = []
    for decision in decisions:
        row = candidates[decision.id]
        if decision.candidate_sha256 != row["candidate_sha256"]:
            raise ValueError("Review refers to a different candidate version")
        # Recompute the content fingerprint, excluding only its added fingerprint/audit fields.
        if canonical_hash({key: value for key, value in row.items() if key not in {"candidate_sha256", "overlap_flagged"}}) != row["candidate_sha256"]:
            raise ValueError("Candidate content changed after generation")
        if decision.decision != "accept":
            continue
        if row["quality_status"] != "machine_passed" or row["overlap_flagged"]:
            raise ValueError("Accepted export requires completed machine checks and no unresolved lexical flag")
        if set(decision.probe_labels) != {probe["kind"] for probe in row["challenge"]["probes"]} or any(label != kind.endswith("pass") for kind, label in decision.probe_labels.items()):
            raise ValueError("Independent probe labels disagree; revise the case before export")
        kept.append(dict(id=row["id"], input=row["case"]["messages"], target=row["case"]["rubric"], metadata=dict(
            section_id=row["section"]["id"], focus_id=row["id"], sections=[row["collection"], row["section"]["id"]],
            top_level_section=row["collection"], source=row["source"], source_sha256=row["source_sha256"],
            candidate_sha256=row["candidate_sha256"], plan_sha256=plan["plan_sha256"],
            review_method=decision.reviewer_type, reviewer=decision.reviewer,
            instruction_role_convention="inspect_system_is_developer")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
    write_json(destination.with_suffix(destination.suffix + ".manifest.json"), dict(rows=len(kept), sha256=sha256(destination),
        plan_sha256=plan["plan_sha256"], decisions_sha256=sha256(decisions_path), generated_at=now(),
        review_methods=sorted({row["metadata"]["review_method"] for row in kept}),
        limitation="Source-grounded curated synthetic evaluations; not OpenAI's human-reviewed benchmark."))
    return kept
