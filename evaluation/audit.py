"""Offline text overlap screening; this is not semantic independence certification."""

from __future__ import annotations

from collections import Counter, defaultdict
import difflib
import json
import math
import re
import unicodedata

from .core import sha256, write_json

STOP = set("a an and are as at be been but by can could did do does for from had has have he her here him his how i if in into is it its just like me my no not of on or our out please s she so some t than that the their them then there these they this those to too us was we were what when where which who why will with would you your".split())


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", normalize(text))


def grams(tokens, n=12):
    return {"\x1f".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def text_fields(message):
    """Screen visible text and the arguments actually sent in tool histories."""
    text = message["content"]
    if not isinstance(text, str):
        raise ValueError("Overlap audit expects textual benchmark inputs")
    if text.strip():
        yield "content", text
    for index, call in enumerate(message.get("tool_calls") or []):
        arguments = call["arguments"]
        yield f"tool_calls[{index}].arguments", (arguments if isinstance(arguments, str) else
            json.dumps(arguments, sort_keys=True, ensure_ascii=False))


def screen_overlap(records, training_rows, sources):
    """Compare every input turn and rubric with every retained training message.

    Return full candidates for human review, never auto-delete benchmark items.
    Delimited source excerpts are screened separately from generated text.
    """
    units, exact_index, span_index = [], defaultdict(set), defaultdict(set)
    for row in training_rows:
        fields = ((index, message, field, text) for index, message in enumerate(row["messages"])
                  for field, text in text_fields(message))
        for index, message, field, text in fields:
            exact_index[normalize(text)].add(row["id"])
            generated = re.sub(r"<excerpt>.*?</excerpt>", " ", text, flags=re.S)
            unit = {"training_id": row["id"], "position": index, "role": message["role"],
                    "field": field, "text": generated, "words": words(generated)}
            units.append(unit)
            for gram in grams(unit["words"]):
                span_index[gram].add(len(units) - 1)
    source_grams = {name: grams(words(text)) for name, text in sources.items()}
    user_units = [u for u in units if u["role"] == "user"]
    terms = [Counter(w for w in u["words"] if w not in STOP) for u in user_units]
    df = Counter(w for counter in terms for w in counter)
    idf = {w: math.log(1 + len(terms) / (1 + n)) for w, n in df.items()}
    postings, lengths = defaultdict(list), []
    for i, counter in enumerate(terms):
        vector = {w: (1 + math.log(n)) * idf[w] for w, n in counter.items()}
        lengths.append(math.sqrt(sum(v * v for v in vector.values())))
        for word, value in vector.items():
            postings[word].append((i, value))

    exact, spans, source_hits, nearest, rubric_hits, containment = [], [], [], [], [], []
    normalized_training = list(exact_index)
    for benchmark, rows in records.items():
        for row in rows:
            identity = {"benchmark": benchmark, "eval_id": row["id"]}
            messages = [{"role": "user", "content": row["input"]}] if isinstance(row["input"], str) else row["input"]
            # Include system/developer/history turns, not just the final user query.
            fields = ((pos, message, field, text) for pos, message in enumerate(messages)
                      for field, text in text_fields(message))
            for pos, message, field, text in fields:
                base = {**identity, "eval_position": pos, "eval_role": message["role"], "eval_field": field, "query": text}
                normalized = normalize(text)
                tokens = words(text)
                if normalized in exact_index:
                    exact.append({**base, "training_ids": sorted(exact_index[normalized])})
                if len(tokens) >= 4:
                    hits = set().union(*(exact_index[t] for t in normalized_training if normalized in t))
                    if hits:
                        containment.append({**base, "training_ids": sorted(hits)})
                query_grams = grams(tokens)
                candidates = set().union(*(span_index[g] for g in query_grams))
                for i in candidates:
                    unit = units[i]
                    match = difflib.SequenceMatcher(None, tokens, unit["words"], autojunk=False).find_longest_match()
                    spans.append({**base, "training_id": unit["training_id"], "training_role": unit["role"],
                                  "training_field": unit["field"],
                                  "shared_words": match.size, "shared_text": " ".join(tokens[match.a:match.a + match.size]),
                                  "training_text": unit["text"]})
                for source, source_set in source_grams.items():
                    if query_grams & source_set:
                        source_hits.append({**base, "source": source, "shared_12grams": len(query_grams & source_set)})
                if message["role"] != "user" or len(tokens) < 4:
                    continue
                vector = {w: (1 + math.log(n)) * idf[w] for w, n in Counter(w for w in tokens if w not in STOP and w in idf).items()}
                length = math.sqrt(sum(v * v for v in vector.values()))
                scores = defaultdict(float)
                for word, value in vector.items():
                    for i, other in postings[word]:
                        scores[i] += value * other
                ranked = sorted(((s / (length * lengths[i]), i) for i, s in scores.items() if length * lengths[i]), reverse=True)[:3]
                for cosine, i in ranked:
                    unit = user_units[i]
                    nearest.append({**base, "training_id": unit["training_id"], "training_text": unit["text"],
                                    "lexical_cosine": cosine,
                                    "token_sequence_ratio": difflib.SequenceMatcher(None, tokens, unit["words"], autojunk=False).ratio()})
            target = row["target"]
            rubric = target if isinstance(target, str) else "\n".join(target)
            normalized_rubric = normalize(rubric)
            if benchmark == "model_spec" and normalized_rubric:
                hits = set().union(*(exact_index[t] for t in normalized_training if normalized_rubric in t))
                if hits:
                    rubric_hits.append({**identity, "training_ids": sorted(hits)})
    nearest.sort(key=lambda x: x["lexical_cosine"], reverse=True)
    spans.sort(key=lambda x: x["shared_words"], reverse=True)
    return {"exact_input_messages": exact, "contained_input_messages_min_4_words": containment,
            "generated_text_12word_matches": spans, "source_12word_matches": source_hits,
            "full_rubric_matches": rubric_hits, "lexical_neighbours": nearest}


def audit(root, records, training_path, output, manual_findings=None):
    previous = output / "overlap_summary.json"
    if manual_findings is not None:
        saved = json.loads(manual_findings.read_text())
        manual = saved if isinstance(saved, list) else [saved]
    else:
        manual = json.loads(previous.read_text()).get("manual_findings", []) if previous.exists() else []
    for hit in manual:
        if not isinstance(hit, dict) or not all(isinstance(hit.get(k), str) and hit[k].strip()
                                               for k in ["eval_id", "kind", "interpretation"]):
            raise ValueError("Manual overlap findings need eval_id, kind and interpretation")
        benchmark = hit.get("benchmark", "model_spec")
        if benchmark not in records or hit["eval_id"] not in {r["id"] for r in records[benchmark]}:
            raise ValueError("Manual overlap finding references an unknown sample")
    rows = [json.loads(line) for line in training_path.read_text().splitlines() if line.strip()]
    sources = {name: (root / name).read_text() for name in ["generic/model_spec.md", "generic/constitution.md"]}
    findings = screen_overlap(records, rows, sources)
    write_json(output / "overlap_candidates.json", findings)
    flagged = defaultdict(set)
    for name in ["exact_input_messages", "contained_input_messages_min_4_words", "generated_text_12word_matches", "full_rubric_matches"]:
        for hit in findings[name]:
            flagged[hit["benchmark"]].add(hit["eval_id"])
    # High lexical similarity is a screening flag, not proof of contamination.
    for hit in findings["lexical_neighbours"]:
        if hit["token_sequence_ratio"] >= 0.8:
            flagged[hit["benchmark"]].add(hit["eval_id"])
    for hit in manual:
        flagged[hit.get("benchmark", "model_spec")].add(hit["eval_id"])
    report = {"training_sha256": sha256(training_path), "training_rows": len(rows),
              "methods": {"normalization": "Unicode NFKC + casefold + whitespace collapse",
                          "input_fields": "Every non-empty message body and serialized tool-call arguments; field paths retained in candidates",
                          "spans": "12 contiguous English word tokens; source <excerpt> spans removed from generated-text search and audited separately",
                          "lexical": "TF-IDF top 3 training user-message neighbours for each input user turn (>=4 words); token sequence ratio >=0.8 flags review",
                          "containment": "Complete evaluation input message (>=4 words) inside any training message; short/generic hits require interpretation"},
              "benchmarks": {}, "manual_findings": manual,
              "flagged_ids": {name: sorted(flagged[name]) for name in records},
              "limitations": ["Lexical screening cannot rule out paraphrased contamination; flagging also produces false positives.",
                              "Source examples shared by synthesis and evals compromise scenario independence without proving eval ingestion.",
                              "Public evaluation exposure in teacher/base pretraining is unknown.",
                              "Official full-set scores remain primary; the flagged-excluded view is auxiliary, not a certified clean test set."]}
    for benchmark in records:
        report["benchmarks"][benchmark] = {"rows": len(records[benchmark]), **{
            key: len({hit["eval_id"] for hit in values if hit["benchmark"] == benchmark})
            for key, values in findings.items()}, "flagged_rows": len(flagged[benchmark])}
        report["benchmarks"][benchmark]["exact_user_messages"] = len({
            hit["eval_id"] for hit in findings["exact_input_messages"]
            if hit["benchmark"] == benchmark and hit["eval_role"] == "user"})
    write_json(output / "overlap_summary.json", report)
    return report
