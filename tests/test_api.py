"""Tests for the initial FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from workrights_assistant.api import AskRequest, app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_question_reaches_not_implemented_endpoint():
    request = AskRequest(question="  How much holiday do I get?  ")
    response = client.post("/ask", json={"question": "  How much holiday do I get?  "})

    assert request.question == "How much holiday do I get?"
    assert response.status_code == 501
    assert response.json() == {"detail": "The ask endpoint is not implemented yet."}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": 123},
        {"question": "a" * 2001},
    ],
    ids=["missing", "empty", "whitespace-only", "wrong-type", "too-long"],
)
def test_invalid_questions_return_validation_error(payload):
    response = client.post("/ask", json=payload)

    assert response.status_code == 422
