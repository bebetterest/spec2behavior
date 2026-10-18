from pathlib import Path

import pytest

from synthesis.markdown_sections import split_sections
from synthesis.portions import split_portions

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("opening,closing", [("````", "`````"), ("~~~~", "~~~~")])
def test_long_fences_keep_embedded_headings_and_short_fences_inside(opening, closing):
    short = opening[:3]
    text = (
        f"# Real\n\n{opening}markdown\n\n{short}python\n# Hidden\n{short}\n\n"
        f"# Still hidden\n{closing}\n\n## Child\n\nVisible body.\n"
    )
    assert [section.title for section in split_sections(text, 1)] == ["Real"]
    assert [section.title for section in split_sections(text, 2)] == ["Child"]
    portions = split_portions(text)
    assert [portion.title for portion in portions] == ["Real", "Child"]
    assert "# Hidden" in portions[0].text and "# Still hidden" in portions[0].text


def test_fenced_headings_are_not_sections():
    text = """# Real {#real}

Intro

~~~xml
```typescript
# functions
not a heading
```
~~~

## Child {#child}

Body of the child.
"""
    top = split_sections(text, 1)
    assert [section.id for section in top] == ["real"]
    assert "# functions" in top[0].body
    child = split_sections(text, 2)
    assert [section.id for section in child] == ["child"]
    assert child[0].ancestors == ("Real",)


def test_duplicate_slugs_get_a_suffix():
    text = "# Notes\n\nFirst.\n\n# Notes\n\nSecond.\n"
    sections = split_sections(text, 1)
    assert [section.id for section in sections] == ["notes", "notes-2"]


def test_model_spec_anchors_ignore_code_fences():
    text = (ROOT / "model_spec" / "model_spec.md").read_text(encoding="utf-8")
    titles = set()
    ids = set()
    for level in (1, 2, 3):
        for section in split_sections(text, level):
            titles.add(section.title)
            ids.add(section.id)
    assert "functions" not in titles
    assert "[...] imports" not in titles
    assert "chain_of_command" in ids
    assert "stay_in_bounds" in ids
    assert "chatgpt_u18" in ids


def test_constitution_sections_have_stable_slugs():
    path = ROOT / "claude-constitution" / "20260120-constitution.md"
    text = path.read_text(encoding="utf-8")
    top = split_sections(text, 1)
    assert "being-helpful" in [section.id for section in top]
    sections = split_sections(text, 2)
    ids = [section.id for section in sections]
    assert any(section_id.startswith("why-helpfulness-is-one-of-claude") for section_id in ids)
    assert len(ids) == len(set(ids))
    assert all(section.body for section in sections)
