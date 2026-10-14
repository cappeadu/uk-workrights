"""Tests for the FastAPI endpoints."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from workrights_assistant import api
from workrights_assistant.api import AskRequest, app

client = TestClient(app)
PILOT_QUESTIONS_FILE = Path(__file__).parent / "fixtures" / "pilot_questions.json"


def get_pilot_question(question_id):
    """Read one question from the reviewed Phase 3 evaluation set."""
    pilot_questions = json.loads(PILOT_QUESTIONS_FILE.read_text(encoding="utf-8"))
    return next(
        question
        for question in pilot_questions["questions"]
        if question["id"] == question_id
    )


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

    set_results(
        [
            {
                "score": 0.5,
                "source_id": "holiday-entitlement",
                "title": "Holiday entitlement",
                "canonical_url": "https://www.gov.uk/holiday-entitlement-rights",
                "text": "Most workers are entitled to paid annual leave.",
            }
        ]
    )
    return set_results


@pytest.fixture
def mock_claude(monkeypatch):
    """Return a fake Claude client and keep a record of requests made to it."""
    calls = []
    response_body = {
        "status": "Answer",
        "answer": "The evidence says workers are entitled to paid annual leave.",
        "source_ids": ["holiday-entitlement"],
    }

    def create(**request):
        calls.append(request)
        response_text = json.dumps(response_body)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=response_text)]
        )

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(api, "get_anthropic_client", lambda: fake_client)
    monkeypatch.setattr(api, "get_anthropic_model", lambda: "test-model")
    return calls, response_body


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_question_with_evidence_calls_claude(mock_retrieval, mock_claude):
    request = AskRequest(question="  How much holiday do I get?  ")
    response = client.post("/ask", json={"question": "  How much holiday do I get?  "})

    assert request.question == "How much holiday do I get?"
    assert response.status_code == 200
    assert response.json() == {
        "status": "answer",
        "answer": "The evidence says workers are entitled to paid annual leave.",
        "citations": [
            {
                "source_id": "holiday-entitlement",
                "title": "Holiday entitlement",
                "excerpt": "Most workers are entitled to paid annual leave.",
                "url": "https://www.gov.uk/holiday-entitlement-rights",
            }
        ],
    }

    calls, _ = mock_claude
    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    assert "How much holiday do I get?" in calls[0]["messages"][0]["content"]
    assert (
        "Most workers are entitled to paid annual leave."
        in calls[0]["messages"][0]["content"]
    )
    assert "only the evidence supplied" in calls[0]["system"]
    assert calls[0]["output_config"]["format"]["schema"] == (
        api.CLAUDE_RESPONSE_SCHEMA
    )


def test_ambiguous_pilot_question_gets_a_cautious_cited_answer(
    mock_retrieval, mock_claude
):
    pilot_question = get_pilot_question("Q07")
    source_id = pilot_question["expected_source_ids"][0]
    mock_retrieval(
        [
            {
                "score": 0.6,
                "source_id": source_id,
                "title": "Dismissal: your rights",
                "canonical_url": "https://www.gov.uk/dismissal",
                "text": "The guidance lists factors relevant to unfair dismissal.",
            }
        ]
    )
    _, response_body = mock_claude
    response_body.update(
        {
            "answer": (
                "The guidance lists factors that may matter, but there is not "
                "enough information here to decide whether this dismissal was unfair."
            ),
            "source_ids": [source_id],
        }
    )

    response = client.post("/ask", json={"question": pilot_question["question"]})

    assert response.status_code == 200
    assert response.json()["status"] == "answer"
    assert "not enough information" in response.json()["answer"]
    assert response.json()["citations"][0]["source_id"] == source_id


@pytest.mark.parametrize(
    "question",
    [
        "How much must my employer contribute to my workplace pension?",
        "What rules apply to workplace pensions?",
        "Can you help me write a CV for a job?",
    ],
    ids=["pension", "pensions", "cv-writing"],
)
def test_clear_out_of_scope_question_returns_out_of_scope(question, mock_claude):
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
    assert mock_claude[0] == []


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
    question, mock_retrieval, mock_claude
):
    response = client.post("/ask", json={"question": question})

    assert response.status_code == 200
    assert response.json()["status"] == "answer"


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


def test_weak_retrieval_score_returns_insufficient_evidence(
    mock_retrieval, mock_claude
):
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
    assert mock_claude[0] == []


def test_no_retrieved_chunks_returns_insufficient_evidence(
    mock_retrieval, mock_claude
):
    mock_retrieval([])

    response = client.post("/ask", json={"question": "What is my employment status?"})

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert mock_claude[0] == []


def test_claude_can_abstain_when_retrieved_evidence_does_not_answer_question(
    mock_retrieval, mock_claude
):
    _, response_body = mock_claude
    response_body.update(
        {
            "status": "insufficient_evidence",
            "answer": "The retrieved passage does not answer this question.",
            "source_ids": [],
        }
    )

    response = client.post("/ask", json={"question": "How much holiday do I get?"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "insufficient_evidence",
        "answer": "The retrieved passage does not answer this question.",
        "citations": [],
    }


def test_insufficient_evidence_can_cite_related_passages(mock_retrieval, mock_claude):
    mock_retrieval(
        [
            {
                "score": 0.5,
                "source_id": "dismissal",
                "title": "Dismissal: your rights",
                "canonical_url": "https://www.gov.uk/dismissal",
                "text": "A union representative can help if you are a union member facing dismissal.",
            }
        ]
    )
    _, response_body = mock_claude
    response_body.update(
        {
            "status": "insufficient_evidence",
            "answer": (
                "The retrieved passage mentions union help with dismissal, but "
                "does not say whether you can join while unemployed."
            ),
            "source_ids": ["dismissal"],
        }
    )

    response = client.post(
        "/ask", json={"question": "Can I join a union if unemployed?"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "insufficient_evidence",
        "answer": (
            "The retrieved passage mentions union help with dismissal, but "
            "does not say whether you can join while unemployed."
        ),
        "citations": [
            {
                "source_id": "dismissal",
                "title": "Dismissal: your rights",
                "excerpt": "A union representative can help if you are a union member facing dismissal.",
                "url": "https://www.gov.uk/dismissal",
            }
        ],
    }


@pytest.mark.parametrize(
    "source_ids",
    [
        ["not-retrieved"],
        ["holiday-entitlement", "not-retrieved"],
    ],
    ids=["unknown-id", "mix-of-valid-and-unknown-ids"],
)
def test_unknown_source_id_replaces_answer_with_abstention(
    source_ids, mock_retrieval, mock_claude
):
    _, response_body = mock_claude
    response_body["answer"] = "This generated claim must not be returned."
    response_body["source_ids"] = source_ids

    response = client.post("/ask", json={"question": "How much holiday do I get?"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "insufficient_evidence",
        "answer": api.UNVERIFIED_CITATION_ANSWER,
        "citations": [],
    }


def test_answer_without_source_ids_becomes_abstention(mock_retrieval, mock_claude):
    _, response_body = mock_claude
    response_body["source_ids"] = []

    response = client.post("/ask", json={"question": "How much holiday do I get?"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "insufficient_evidence",
        "answer": api.UNVERIFIED_CITATION_ANSWER,
        "citations": [],
    }


def test_duplicate_source_ids_use_one_highest_ranked_excerpt(
    mock_retrieval, mock_claude
):
    mock_retrieval(
        [
            {
                "score": 0.9,
                "source_id": "holiday-entitlement",
                "title": "Holiday entitlement",
                "canonical_url": "https://www.gov.uk/holiday-entitlement-rights",
                "text": "The highest-ranked passage.",
            },
            {
                "score": 0.8,
                "source_id": "holiday-entitlement",
                "title": "Holiday entitlement",
                "canonical_url": "https://www.gov.uk/holiday-entitlement-rights",
                "text": "A lower-ranked passage.",
            },
        ]
    )
    _, response_body = mock_claude
    response_body["source_ids"] = ["holiday-entitlement", "holiday-entitlement"]

    response = client.post("/ask", json={"question": "How much holiday do I get?"})

    assert response.status_code == 200
    citations = response.json()["citations"]
    assert len(citations) == 1
    assert citations[0]["excerpt"] == "The highest-ranked passage."


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
