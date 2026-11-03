"""Strict request/response contracts for evaluation construction."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints, field_validator, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class Source(Record):
    name: Text
    path: Text
    sections: list[Text] = Field(default_factory=list)

    @field_validator("sections")
    @classmethod
    def valid_patterns(cls, values):
        import re
        for value in values:
            try:
                re.compile(value)
            except re.error as error:
                raise ValueError(f"Invalid section regex: {error}") from error
        return values

class Settings(Record):
    sources: list[Source] = Field(min_length=1)
    training_data: list[Text] = Field(min_length=1)
    provider: Literal["deepseek", "openrouter", "openai"] = "deepseek"
    base_url: Text = "https://api.deepseek.com"
    api_key_env: Text = "DEEPSEEK_API_KEY"
    model: Text = "deepseek-flash"
    stage_models: dict[Literal["author", "reviewer", "judge"], Text] = Field(default_factory=dict)
    thinking: bool | None = True
    temperature: FiniteFloat = Field(default=1.0, ge=0, le=2)
    max_tokens: int = Field(default=131072, ge=1024)
    timeout: FiniteFloat = Field(default=1800.0, gt=0)
    concurrency: int = Field(default=6, ge=1, le=128)
    max_sections_per_source: int = Field(default=2, ge=1)
    max_revisions: int = Field(default=1, ge=0, le=3)
    max_calls: int = Field(default=96, ge=1)
    max_estimated_usd: FiniteFloat = Field(default=5.0, gt=0)
    input_miss_usd_per_million: FiniteFloat = Field(default=0.30, ge=0)
    input_hit_usd_per_million: FiniteFloat = Field(default=0.006, ge=0)
    output_usd_per_million: FiniteFloat = Field(default=1.20, ge=0)
    pricing_source: Text = "https://api-docs.deepseek.com/quick_start/pricing/"
    pricing_checked: Text = "2026-10-25"
    grading_mode: Literal["rubric-only", "full-spec"] = "rubric-only"

    @model_validator(mode="after")
    def check_sources(self):
        from urllib.parse import urlsplit
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) API root without credentials, query or fragment")
        if len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("source names must be unique")
        if self.provider == "openai" and self.thinking is not None:
            raise ValueError("For generic OpenAI-compatible endpoints set thinking to null")
        return self

class Span(Record):
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    quote: Text

class Focus(Record):
    rule: Text
    strength: Literal["must", "should", "may", "aspirational", "conditional"]
    applicability: Text
    exceptions: Text
    priority: Text
    identity_dependent: bool
    evidence: list[Span] = Field(min_length=1, max_length=5)

class Extraction(Record):
    focus: Focus | None
    omissions: list[Text]

class Review(Record):
    verdict: Literal["accept", "revise", "reject"]
    explanation: Text
    issues: list[Text]
    evidence: list[Span] = Field(min_length=1, max_length=5)

class Message(Record):
    role: Literal["system", "user", "assistant"]
    content: Text

class Case(Record):
    title: Text
    messages: list[Message] = Field(min_length=1, max_length=12)
    rubric: Text

    @model_validator(mode="after")
    def conversation(self):
        expected = "user"
        started = False
        for message in self.messages:
            if message.role in {"system", "developer"}:
                if started:
                    raise ValueError("instruction messages must precede the dialogue")
            else:
                if message.role != expected:
                    raise ValueError("dialogue must alternate user/assistant")
                started = True
                expected = "assistant" if expected == "user" else "user"
        if not started or self.messages[-1].role != "user":
            raise ValueError("input must end before the final answer being evaluated")
        return self

class Probe(Record):
    kind: Literal["minimal_pass", "alternate_pass", "near_miss_fail", "clear_fail"]
    response: Text
    basis: Text

class Challenge(Record):
    probes: list[Probe] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def coverage(self):
        if {probe.kind for probe in self.probes} != {"minimal_pass", "alternate_pass", "near_miss_fail", "clear_fail"}:
            raise ValueError("exactly one probe of each required kind is needed")
        if len({probe.response for probe in self.probes}) != 4:
            raise ValueError("probe responses must differ")
        return self

class Grade(Record):
    section_id: Text
    compliance: float = Field(ge=1, le=7)
    critique: Text

class Decision(Record):
    id: Text
    candidate_sha256: Text
    decision: Literal["accept", "reject", "uncertain"]
    reviewer: Text
    reviewer_type: Literal["human", "codex"]
    rationale: Text
    probe_labels: dict[Literal["minimal_pass", "alternate_pass", "near_miss_fail", "clear_fail"], bool]
