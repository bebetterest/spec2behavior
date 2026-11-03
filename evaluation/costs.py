"""Uncached token-based estimates using the actual upstream judge prompts."""

from __future__ import annotations

from copy import deepcopy
from statistics import mean

from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelName, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from tinker_cookbook import renderers
from tinker_cookbook.eval.inspect_utils import convert_inspect_messages

from .provider import DeveloperRoleRenderer, judge_messages


def input_messages(sample):
    return [ChatMessageUser(content=sample.input)] if isinstance(sample.input, str) else deepcopy(sample.input)


def pilot_ids(tasks, minimum):
    """Deterministic coverage of metadata groups plus role/tool-history canaries."""
    selected = {}
    for name, task in tasks.items():
        key = {"model_spec": "top_level_section", "xstest_safe": "type",
               "xstest_unsafe": "type", "strong_reject": "category"}[name]
        seen, ids = set(), []
        for sample in task.dataset:
            value = (sample.metadata or {}).get(key)
            if value not in seen:
                ids.append(sample.id)
                seen.add(value)
        if name == "model_spec":
            for sample in task.dataset:
                if any(m.role == "tool" or getattr(m, "tool_calls", None) for m in input_messages(sample)):
                    if sample.id not in ids:
                        ids.append(sample.id)
            for role in ["system", "assistant"]:
                sample = next(s for s in task.dataset if any(m.role == role for m in input_messages(s)))
                if sample.id not in ids:
                    ids.append(sample.id)
        for sample in task.dataset:
            if len(ids) >= minimum:
                break
            if sample.id not in ids:
                ids.append(sample.id)
        selected[name] = ids
    return selected


def estimate_costs(token_counts, config, profiles, target_count=2):
    if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count < 1:
        raise ValueError("target_count must be a positive integer")
    prices = config["pricing"]
    scenarios = list(prices["scenarios"]) + [{
        "name": "at_output_limits_estimate", "target_total_output_tokens": config["sampling"]["max_tokens"],
        "target_final_answer_tokens": config["sampling"]["max_tokens"],
        "judge_total_output_tokens": config["judge"]["max_tokens"]}]
    output = {}
    for profile, selected in profiles.items():
        benchmarks = {}
        for name, ids in selected.items():
            rows = [r for r in token_counts[name] if r["id"] in set(ids)]
            if len(rows) != len(ids):
                raise ValueError("Cost profile contains duplicate or missing sample ids")
            benchmarks[name] = {"samples_per_model": len(rows),
                "target_input_tokens_all_models": target_count * sum(r["target_input_tokens"] for r in rows),
                "judge_fixed_input_tokens_all_models": target_count * sum(r["judge_fixed_input_tokens"] for r in rows)}
        estimates = []
        for scenario in scenarios:
            scenario = {**scenario,
                "target_total_output_tokens": min(scenario["target_total_output_tokens"], config["sampling"]["max_tokens"]),
                "target_final_answer_tokens": min(scenario["target_final_answer_tokens"], config["sampling"]["max_tokens"]),
                "judge_total_output_tokens": min(scenario["judge_total_output_tokens"], config["judge"]["max_tokens"])}
            per_benchmark = {}
            for name, counts in benchmarks.items():
                calls = target_count * counts["samples_per_model"]
                target = (counts["target_input_tokens_all_models"] * prices["target_input_usd_per_million"]
                          + calls * scenario["target_total_output_tokens"] * prices["target_output_usd_per_million"]) / 1e6
                judge = ((counts["judge_fixed_input_tokens_all_models"] + calls * scenario["target_final_answer_tokens"]) * prices["judge_input_usd_per_million"]
                         + calls * scenario["judge_total_output_tokens"] * prices["judge_output_usd_per_million"]) / 1e6
                per_benchmark[name] = {"target_usd": target, "judge_usd": judge, "total_usd": target + judge}
            total = sum(x["total_usd"] for x in per_benchmark.values())
            estimates.append({**scenario, "benchmarks": per_benchmark, "total_usd": total,
                              "with_review_contingency_usd": total * (1 + prices["contingency_fraction"])})
        output[profile] = {"target_count": target_count, "benchmarks": benchmarks,
                          "target_calls": sum(target_count * x["samples_per_model"] for x in benchmarks.values()),
                          "judge_calls_max": sum(target_count * x["samples_per_model"] for x in benchmarks.values()), "scenarios": estimates}
    return output


async def measure_prompts(tasks, judge, tokenizer, renderer_name, judge_renderer):
    marker = "Offline candidate answer for prompt measurement."
    result = {}
    for name, task in tasks.items():
        renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        if name == "model_spec":
            renderer = DeveloperRoleRenderer(renderer)
        rows = []
        for sample in task.dataset:
            messages = input_messages(sample)
            target_messages = list(messages)
            if name.startswith("xstest"):
                from inspect_ai.model import ChatMessageSystem
                target_messages.insert(0, ChatMessageSystem(content="You are a helpful assistant."))
            target_tokens = len(renderer.build_generation_prompt(convert_inspect_messages(target_messages)).to_ints())
            candidate = ChatMessageAssistant(content=marker)
            state = TaskState(model=ModelName(judge), sample_id=sample.id, epoch=1,
                              input=sample.input, messages=messages + [candidate], metadata=sample.metadata,
                              output=ModelOutput.from_content("offline-candidate", marker))
            before = len(judge.api.calls)
            await task.scorer[0](state, Target(sample.target))
            if len(judge.api.calls) != before + 1:
                raise ValueError("Expected exactly one upstream judge call per non-empty reply")
            captured, judge_config = judge.api.calls[-1]
            converted = convert_inspect_messages(judge_messages(captured, judge_config))
            prompt = judge_renderer.build_generation_prompt(converted)
            fixed_tokens = len(prompt.to_ints()) - len(judge_renderer.tokenizer.encode(marker, add_special_tokens=False))
            rows.append({"id": sample.id, "target_input_tokens": target_tokens, "judge_fixed_input_tokens": fixed_tokens})
            judge.api.calls.clear()
        result[name] = rows
    return result


def cost_report(counts, config, selected, target_count=2):
    profiles = {"pilot": selected, "full": {k: [r["id"] for r in rows] for k, rows in counts.items()}}
    return {"pricing": config["pricing"], "model_spec_grading": config["judge"]["model_spec_grading"],
            "profiles": estimate_costs(counts, config, profiles, target_count),
            "token_summary": {name: {key: {"mean": mean(r[key] for r in rows), "max": max(r[key] for r in rows)}
                                      for key in ["target_input_tokens", "judge_fixed_input_tokens"]} for name, rows in counts.items()},
            "assumptions": [f"{target_count} targets; every non-empty sample has one judge call; all input charged at uncached standard rates.",
                            f"Model Spec judge mode: {config['judge']['model_spec_grading']}. Captured judge inputs use the selected grading prompt.",
                            "Target prompts use the frozen training tokenizer and renderer; judge prompts are captured from upstream scorers offline.",
                            "Judge inputs use the frozen native tokenizer/renderer, fixed system date, tool history and the same schema-format instruction as execution; no OpenAI API token approximation.",
                            "Target output lengths include thinking and final text; final-answer lengths in scenarios are assumed judge-tokenizer tokens. Scenario lengths are clamped to configured caps.",
                            "The at-output-limits case assumes final-answer judge tokens equal the target token cap; tokenizer differences mean it is not a hard spending ceiling.",
                            f"The {config['pricing']['contingency_fraction']:.0%} contingency is a planning allowance, not permission to retry or increase caps; automatic generation retries are disabled.",
                            "No cache/batch/promotion beyond the stated Tinker rate is assumed. Existing checkpoint storage is excluded.",
                            "Tinker tariffs must be rechecked before execution; a different judge requires updating the judge model/renderer/context and its two unit prices."]}
