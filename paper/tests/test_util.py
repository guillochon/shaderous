"""Unit tests for the TeX helpers and the pure parts of the util/ scripts. No TeX install needed."""

import check_citations
import paperconf
import pytest
from arxiv_zip import drop_comment_text

import edits


def test_split_comment_keeps_escaped_percent():
    assert paperconf.split_comment(r"5\% of stars % a note") == (r"5\% of stars ", "% a note")
    assert paperconf.split_comment(r"a \\% comment") == ("a \\\\", "% comment")


def test_drop_comment_text_keeps_percent_and_drops_comment_lines():
    tex = "a % note\n% whole line\nb\\%c\n"
    assert drop_comment_text(tex) == "a %\nb\\%c\n"


def test_abstract_environment_and_structured():
    assert paperconf.abstract("\\begin{abstract}\n  One   two.\n\\end{abstract}") == "One two."
    aa = r"\abstract{Context.}{Aims.}{}{Results {with braces}.}{}"
    assert paperconf.abstract(aa) == "Context. Aims. Results {with braces}."


def test_title_with_short_form_and_nested_braces():
    assert paperconf.title(r"\title[Short]{A {Nested} Title}") == "A {Nested} Title"


def test_preamble_macros():
    tex = r"\newcommand{\Msun}{M}\providecommand\foo{x}\def\bar{y}\begin{document}\newcommand{\late}{z}"
    assert paperconf.preamble_macros(tex) == {"Msun", "foo", "bar"}


def test_parse_bib_fields():
    fields = check_citations.parse_fields(' author = {{Hogg}, D. and {Bovy}, J.}, title = "{A {B} C}", year = 2010,')
    assert fields["year"] == "2010"
    assert check_citations.first_author(fields) == "Hogg"
    assert fields["title"] == "{A {B} C}"


def test_compare_flags_wrong_year_and_author():
    fields = {"title": "Ensemble samplers", "year": "2011", "author": "{Goodman}, J."}
    found = {"title": "Ensemble samplers", "year": 2010, "first_author": "Weare"}
    issues = check_citations.compare(fields, found)
    assert any(i.startswith("year") for i in issues)
    assert any(i.startswith("AUTHOR") for i in issues)


def test_edits_sub_requires_unique_match():
    with pytest.raises(SystemExit):
        edits.apply_edit("a a", ("sub", "a", "b"))
    assert edits.apply_edit("one two", ("sub", "two", "three"))[0] == "one three"


def test_edits_para_and_span():
    text = "first line\n\nsecond line\nthird line\n"
    assert edits.apply_edit(text, ("para", "first", None))[0] == "second line\nthird line\n"
    assert edits.apply_edit(text, ("span", "second", "third", "merged"))[0] == "first line\n\nmerged\n"


def test_template_renders_every_journal():
    import new_paper

    conf = {
        "paper": {"slug": "t", "title": "T", "keywords": ["a", "b"], "bibs": ["references.bib"]},
        "authors": [
            {"name": "Ada Lovelace", "email": "a@x", "affiliation": "One", "corresponding": True},
            {"name": "Carl Gauss", "affiliation": "Two"},
        ],
    }
    for journal in new_paper.journals():
        out = new_paper.render(journal, conf, "T", "Abstract.", "Body.")
        assert "<<" not in out
        assert "%% ---- BODY ----\nBody.\n%% ---- END BODY ----" in out
