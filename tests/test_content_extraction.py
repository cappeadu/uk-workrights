from scripts.extract_sources import extract_source, html_to_markdown


def test_html_to_markdown_preserves_headings_paragraphs_and_links():
    html = (
        "<h2>Eligibility</h2><p>See <a href='/holiday-entitlement-rights'>"
        "holiday guidance</a> &amp; check the details.</p>"
    )

    markdown = html_to_markdown(html, "https://www.gov.uk/statutory-sick-pay")

    assert "## Eligibility" in markdown
    assert "[holiday guidance](https://www.gov.uk/holiday-entitlement-rights)" in markdown
    assert "& check the details." in markdown


def test_html_to_markdown_preserves_nested_lists_and_tables():
    html = """
    <ul><li>First<li>Second<ul><li>Nested</ul></ul>
    <table><tr><th>Day</th><th>Leave</th></tr>
    <tr><td>Monday</td><td>Paid</td></tr></table>
    """

    markdown = html_to_markdown(html, "https://www.gov.uk/holiday-entitlement-rights")

    assert "- First" in markdown
    assert "- Second" in markdown
    assert "  - Nested" in markdown
    assert "| Day | Leave |" in markdown
    assert "| --- | --- |" in markdown
    assert "| Monday | Paid |" in markdown


def test_extract_source_keeps_part_titles_and_source_identity():
    source = {
        "id": "example-guide",
        "title": "Example guide",
        "canonical_path": "/example-guide",
        "content_id": "content-123",
        "document_type": "guide",
        "schema_name": "guide",
    }
    item = {
        "title": "Example guide",
        "base_path": "/example-guide",
        "content_id": "content-123",
        "document_type": "guide",
        "schema_name": "guide",
        "details": {"parts": [{"title": "Part one", "body": "<p>Useful guidance.</p>"}]},
    }

    markdown = extract_source(item, source)

    assert "# Example guide" in markdown
    assert "Source ID: `example-guide`" in markdown
    assert "Content ID: `content-123`" in markdown
    assert "## Part one" in markdown
    assert "Useful guidance." in markdown
