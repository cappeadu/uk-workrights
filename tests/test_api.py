"""Tests for the initial FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from workrights_assistant import api
from workrights_assistant.api import AskRequest, app

client = TestClient(app)


@pytest.fixture
def mock_retrieval(monkeypatch):
    """Avoid loading the real embedding model or Qdrant in endpoint tests."""
    monkeypatch.delenv(api.EVIDENCE_SCORE_THRESHOLD_ENV, raising=False)
    monkeypatch.setattr(api, "get_qdrant_client", lambda: object())
    monkeypatch.setattr(api, "get_embedding_model", lambda: object())

    def set_results(results):
        monkeypatch.setattr(
            api,
            "retrieve_top_five",
            lambda question, client, model, collection: results,
        )

    set_results([{"score": 0.5}])
    return set_results


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_question_with_evidence_reaches_not_implemented_endpoint(mock_retrieval):
    request = AskRequest(question="  How much holiday do I get?  ")
    response = client.post("/ask", json={"question": "  How much holiday do I get?  "})

    assert request.question == "How much holiday do I get?"
    assert response.status_code == 501
    assert response.json() == {"detail": "The ask endpoint is not implemented yet."}


@pytest.mark.parametrize(
    "question",
    [
        "How much must my employer contribute to my workplace pension?",
        "What rules apply to workplace pensions?",
        "Can you help me write a CV for a job?",
    ],
    ids=["pension", "pensions", "cv-writing"],
)
def test_clear_out_of_scope_question_returns_out_of_scope(question):
    response = client.post("/ask", json={"question": question})

    assert response.status_code == 200
    assert response.json() == {
        "status": "out_of_scope",
        "answer": (
            "I can help with selected employment topics covered by GOV.UK. "
            "This question is outside the current topic scope."
        ),
        "citations": [],
    }


@pytest.mark.parametrize(
    "question",
    [
        "What does my employment contract say about pension contributions?",
        "What does compassionate leave cover?",
        "How much paid holiday do I get?",
    ],
    ids=["mixed-topics", "unknown-topic", "in-scope-topic"],
)
def test_unclear_or_in_scope_question_is_not_classified_out_of_scope(
    question, mock_retrieval
):
    response = client.post("/ask", json={"question": question})

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


def test_weak_retrieval_score_returns_insufficient_evidence(mock_retrieval):
    mock_retrieval([{"score": 0.34}])

    response = client.post("/ask", json={"question": "What is my employment status?"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "insufficient_evidence",
        "answer": (
            "I could not find enough relevant information in the selected "
            "GOV.UK sources to answer that question."
        ),
        "citations": [],
    }


def test_no_retrieved_chunks_returns_insufficient_evidence(mock_retrieval):
    mock_retrieval([])

    response = client.post("/ask", json={"question": "What is my employment status?"})

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"


def test_evidence_threshold_can_be_changed_with_environment_variable(
    monkeypatch, mock_retrieval
):
    monkeypatch.setenv(api.EVIDENCE_SCORE_THRESHOLD_ENV, "0.6")
    mock_retrieval([{"score": 0.5}])

    response = client.post("/ask", json={"question": "What is my employment status?"})

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"


def test_invalid_evidence_threshold_returns_configuration_error(
    monkeypatch, mock_retrieval
):
    monkeypatch.setenv(api.EVIDENCE_SCORE_THRESHOLD_ENV, "2")

    response = client.post("/ask", json={"question": "What is my employment status?"})

    assert response.status_code == 500
    assert api.EVIDENCE_SCORE_THRESHOLD_ENV in response.json()["detail"]
