"""Conservative source citation normalization; preserve raw API records."""
import re

_NOTE = re.compile(r"\[\^[^\]\n]+\]")


def resolve_citation_markers(span, source):
    """Only restore Markdown footnote markers, inside the supplied line range.

    Every non-marker character must match, uniquely. Words, punctuation,
    modality, exceptions, and line coordinates are never guessed or rewritten.
    """
    lines = source.splitlines()
    if not 1 <= span.line_start <= span.line_end <= len(lines):
        raise ValueError("Source line range is out of bounds")
    window = "\n".join(lines[span.line_start - 1:span.line_end])
    if span.quote in window:
        return None
    cleaned = []; mapping = []; position = 0
    while position < len(window):
        marker = _NOTE.match(window, position)
        if marker:
            position = marker.end(); continue
        cleaned.append(window[position]); mapping.append(position); position += 1
    clean_window = "".join(cleaned)
    clean_quote = _NOTE.sub("", span.quote)
    if not clean_quote or clean_window.count(clean_quote) != 1:
        raise ValueError("Source quote mismatch is not a unique citation-marker omission")
    start = clean_window.index(clean_quote)
    resolved = window[mapping[start]:mapping[start + len(clean_quote) - 1] + 1]
    original = span.quote
    span.quote = resolved
    if span.quote not in window:
        raise ValueError("Citation-marker normalization did not match source")
    return dict(kind="citation_marker_normalization", supplied_quote=original,
                resolved_quote=resolved, line_start=span.line_start, line_end=span.line_end)
