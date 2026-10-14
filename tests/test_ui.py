"""Tests for the Streamlit response display."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

UI_FILE = Path(__file__).parents[1] / "ui" / "app.py"


def run_ui(response=None):
    """Run the UI with an optional current API response."""
    app = AppTest.from_file(str(UI_FILE))
    if response is not None:
        app.session_state["response"] = response
    return app.run()


def test_ui_shows_question_input_and_empty_state():
    app = run_ui()

    assert not app.exception
    assert app.chat_input[0].placeholder == "Ask an employment question"
    assert app.caption[0].value == "General information from GOV.UK. This service is not legal advice."


@pytest.mark.parametrize(
    ("status", "element", "label"),
    [
        ("answer", "success", "Answer"),
        ("insufficient_evidence", "warning", "Not enough evidence"),
        ("out_of_scope", "info", "Outside the current topic scope"),
    ],
)
def test_ui_shows_the_response_status(status, element, label):
    app = run_ui({"status": status, "answer": "Example response.", "citations": []})

    assert not app.exception
    assert getattr(app, element)[0].value == label


def test_ui_displays_citation_excerpt_and_official_link():
    response = {
        "status": "answer",
        "answer": "Workers are entitled to paid annual leave.",
        "citations": [
            {
                "source_id": "holiday-entitlement",
                "title": "Holiday entitlement",
                "excerpt": "Most workers are entitled to paid annual leave.",
                "url": "https://www.gov.uk/holiday-entitlement-rights",
            }
        ],
    }

    app = run_ui(response)

    assert not app.exception
    assert app.expander[0].label == "Sources (1)"
    assert app.link_button[0].label == "Open GOV.UK source"
