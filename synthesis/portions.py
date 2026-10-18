"""Split a spec or constitution into heading-based rule excerpts.

The direct body under each heading stays together when it fits the character
limit. Longer bodies are grouped on paragraph boundaries, and oversized
paragraphs are cut near a newline or space. A one-line lead-in stays with the
paragraph it introduces. A short tail of an oversized paragraph stays with
the previous piece. Headings inside fenced code are ignored. A block that is
only a table of contents is skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from synthesis.markdown_sections import (
    _closes_fence,
    _fence_marker,
    _headings_outside_fences,
    _slug,
    _unique_id,
)

_TOC_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)])?\s*\[[^\]]+\]\(#[^)]*\)\s*$")
_EXAMPLE_LABEL = re.compile(r"^(?:\*\*)?Example(?:\*\*)?\s*:")
_BOLD_LABEL = re.compile(r"^\*\*[^*]+\*\*$")

DEFAULT_MAX_PORTION_CHARS = 16000


@dataclass(frozen=True)
class Portion:
    """One piece of rule text.

    ``id`` is ``<heading id>/<index>`` inside a single file. The template adds
    the filename when it builds the job id.
    """

    id: str
    title: str
    ancestors: tuple[str, ...]
    text: str


def split_portions(text: str, max_chars: int = DEFAULT_MAX_PORTION_CHARS) -> list[Portion]:
    """Return excerpts in document order.

    Text before the first heading is omitted. The body of a heading stops at
    the next heading of any level, so a child section is not repeated in its
    parent. Paragraphs in that direct body stay together unless it is too
    long for ``max_chars``.
    """

    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    headings = _headings_outside_fences(text)
    lines = text.splitlines()
    seen: dict[str, int] = {}
    stack: list[tuple[int, str]] = []
    portions: list[Portion] = []
    for index, (line_index, level, title, anchor) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        ancestors = tuple(item[1] for item in stack)
        stack.append((level, title))
        end = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_index + 1 : end]).strip()
        base = _unique_id(anchor or _slug(title), seen)
        if not body:
            continue
        paragraphs = [paragraph for paragraph in _paragraphs(body) if not _is_toc(paragraph)]
        for piece_index, piece in enumerate(_section_pieces(_merge_lead_ins(paragraphs), max_chars)):
            portions.append(
                Portion(
                    id=f"{base}/{piece_index}",
                    title=title,
                    ancestors=ancestors,
                    text=piece,
                )
            )
    return portions


def _section_pieces(paragraphs: list[str], limit: int) -> list[str]:
    """Keep a direct body together, using paragraph boundaries when it is long."""

    parts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + 2 + len(paragraph) <= limit:
            current = f"{current}\n\n{paragraph}"
            continue
        if current:
            parts.append(current)
        pieces = _fit(paragraph, limit)
        parts.extend(pieces[:-1])
        current = pieces[-1]
    if current:
        parts.append(current)
    return parts


def _paragraphs(body: str) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    fence: str | None = None
    for line in body.splitlines():
        marker = _fence_marker(line)
        if fence is not None:
            buf.append(line)
            if _closes_fence(line, fence):
                fence = None
            continue
        if marker is not None:
            fence = marker
            buf.append(line)
            continue
        if not line.strip():
            if buf:
                chunks.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        chunks.append("\n".join(buf).strip())
    return [chunk for chunk in chunks if chunk]


def _is_toc(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and all(_TOC_LINE.match(line) for line in lines)


def _is_lead_in(text: str) -> bool:
    """A label that introduces the next paragraph, not a rule on its own."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    last = lines[-1]
    if len(lines) == 1 and (
        last.startswith("!!!") or _EXAMPLE_LABEL.match(last) or _BOLD_LABEL.match(last)
    ):
        return True
    return last.endswith(":")


def _merge_lead_ins(paragraphs: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(paragraphs):
        chunk = paragraphs[index]
        index += 1
        while index < len(paragraphs) and _is_lead_in(chunk):
            chunk = f"{chunk}\n\n{paragraphs[index]}"
            index += 1
        merged.append(chunk)
    return merged


def _fit(text: str, limit: int) -> list[str]:
    """Cut near a newline or space. A leftover shorter than half the limit stays put."""

    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = max(window.rfind("\n"), window.rfind(" "))
        if cut < limit // 2:
            cut = limit
        if len(rest) - cut < limit // 2:
            break
        piece = rest[:cut].strip()
        nxt = rest[cut:].strip()
        if not piece or not nxt or nxt == rest:
            break
        parts.append(piece)
        rest = nxt
    if rest:
        parts.append(rest)
    return parts
