from pathlib import Path

import pytest

from synthesis.portions import DEFAULT_MAX_PORTION_CHARS, split_portions

ROOT = Path(__file__).resolve().parents[1]


def test_child_text_is_not_repeated_and_preamble_is_omitted():
    text = (
        "Preamble stays out.\n\n# Parent {#parent}\n\nParent only.\n\n"
        "More parent context.\n\n## Child {#child}\n\nChild only.\n\n"
        "More child context.\n"
    )
    portions = split_portions(text)
    assert [(item.id, item.title, item.ancestors, item.text) for item in portions] == [
        ("parent/0", "Parent", (), "Parent only.\n\nMore parent context."),
        ("child/0", "Child", ("Parent",), "Child only.\n\nMore child context."),
    ]


def test_fenced_heading_is_not_a_portion():
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
    portions = split_portions(text)
    assert [item.title for item in portions] == ["Real", "Child"]
    assert portions[0].text.startswith("Intro\n\n~~~xml")
    assert portions[0].id == "real/0"
    assert "# functions" in portions[0].text
    assert portions[1].id == "child/0"
    assert portions[1].text == "Body of the child."
    assert portions[1].ancestors == ("Real",)


def test_toc_is_skipped_and_empty_headings_keep_later_ids_stable():
    text = """# Notes

- [One](#one)
* [Two](#two)

# Notes

The real note.
"""
    portions = split_portions(text)
    assert [(item.id, item.text) for item in portions] == [("notes-2/0", "The real note.")]


def test_long_paragraph_splits_on_spaces_and_keeps_a_short_tail():
    words = ("alpha " * 40).strip()
    portions = split_portions("# Rule {#rule}\n\n" + words + "\n", max_chars=50)
    assert [item.id for item in portions] == [f"rule/{index}" for index in range(len(portions))]
    assert " ".join(item.text for item in portions) == words
    assert all(len(item.text) <= 50 for item in portions[:-1])
    assert len(portions[-1].text) < 75

    hard = split_portions("# Rule\n\n" + ("x" * 100), max_chars=30)
    assert [item.text for item in hard] == ["x" * 30, "x" * 30, "x" * 40]
    assert [item.id for item in hard] == ["rule/0", "rule/1", "rule/2"]

    paragraphs = ["Keep the first rule.", "Keep the second rule.", "Keep the third rule."]
    grouped = split_portions("# Rule\n\n" + "\n\n".join(paragraphs), max_chars=45)
    assert [item.text for item in grouped] == ["\n\n".join(paragraphs[:2]), paragraphs[2]]
    assert [item.id for item in grouped] == ["rule/0", "rule/1"]


def test_lead_in_stays_with_the_following_paragraph():
    text = """# Rule {#rule}

These aims can conflict:

- Help people.
- Avoid harm.

**Example**: tooth fairy

~~~xml
<user>Is it real?</user>
~~~

!!! meta "Commentary"

    This note is the commentary.

**Types of uncertainty**

The assistant may not know.

A complete sentence stands alone.

Another sentence.
"""
    portions = split_portions(text, max_chars=100)
    assert portions[0].text.startswith("These aims can conflict:")
    assert "Avoid harm." in portions[0].text
    assert portions[1].text.startswith("**Example**: tooth fairy")
    assert "<user>Is it real?</user>" in portions[1].text
    assert portions[2].text.startswith('!!! meta "Commentary"')
    assert "This note is the commentary." in portions[2].text
    assert portions[3].text.startswith("**Types of uncertainty**")
    assert "The assistant may not know." in portions[3].text
    assert "\n\n".join(item.text for item in portions).endswith(
        "A complete sentence stands alone.\n\nAnother sentence."
    )


def test_max_chars_must_be_positive():
    with pytest.raises(ValueError):
        split_portions("# Rule\n\nText.\n", max_chars=0)


def test_generic_documents_split_into_bounded_unique_portions():
    for name in ("model_spec.md", "constitution.md"):
        text = (ROOT / "generic" / name).read_text(encoding="utf-8")
        portions = split_portions(text)
        assert portions
        assert len({item.id for item in portions}) == len(portions)
        assert all(0 < len(item.text) < DEFAULT_MAX_PORTION_CHARS * 1.5 for item in portions)
        assert "functions" not in {item.title for item in portions}
        assert not any(item.text.strip() in {'!!! meta "Commentary"', "These aims can conflict:"} for item in portions)
        assert not any(
            item.text.startswith("**Example**:") and "\n" not in item.text and len(item.text) < 80
            for item in portions
        )
