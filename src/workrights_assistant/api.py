"""FastAPI endpoints for the Worker Employment Information Assistant."""

import os
import re
from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints
from qdrant_client import QdrantClient

from workrights_assistant.retrieval import (
    DEFAULT_COLLECTION,
    DEFAULT_STORAGE,
    load_embedding_model,
    retrieve_top_five,
)

app = FastAPI(title="Worker Employment Information Assistant")

# This threshold keeps the lowest-scoring supported pilot question above the
# default. It is a relevance filter, not proof that retrieved text answers a question.
DEFAULT_EVIDENCE_SCORE_THRESHOLD = 0.35
EVIDENCE_SCORE_THRESHOLD_ENV = "EVIDENCE_SCORE_THRESHOLD"

OUT_OF_SCOPE_TERMS = (
    "pension",
    "pensions",
    "cv",
    "resume",
    "résumé",
    "cover letter",
    "job search",
    "job application",
    "find a job",
    "looking for a job",
    "payroll",
    "hiring",
    "recruitment",
    "hire staff",
    "hire employees",
    "recruit staff",
    "recruit employees",
)

IN_SCOPE_TERMS = (
    "employment status",
    "employee status",
    "contract",
    "holiday",
    "annual leave",
    "sick",
    "statutory sick pay",
    "ssp",
    "working hours",
    "hours",
    "redundancy",
    "dismissal",
    "maternity",
)


class AskRequest(BaseModel):
    """A validated question submitted to the assistant."""

    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=2000,
        ),
    ]


def contains_any_term(question: str, terms: tuple[str, ...]) -> bool:
    """Check for whole words or phrases, ignoring case and extra whitespace."""
    normalized_question = " ".join(question.casefold().split())
    return any(
        re.search(rf"\b{re.escape(term)}\b", normalized_question) for term in terms
    )


def is_out_of_scope(question: str) -> bool:
    """Flag clear excluded topics unless the question also mentions an MVP topic."""
    mentions_excluded_topic = contains_any_term(question, OUT_OF_SCOPE_TERMS)
    mentions_in_scope_topic = contains_any_term(question, IN_SCOPE_TERMS)

    return mentions_excluded_topic and not mentions_in_scope_topic


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Open local Qdrant storage on the first in-scope question."""
    return QdrantClient(path=str(DEFAULT_STORAGE))


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the local CPU embedding model on the first in-scope question."""
    return load_embedding_model()


def get_evidence_score_threshold() -> float:
    """Read and validate the optional Qdrant cosine-score threshold."""
    raw_threshold = os.getenv(
        EVIDENCE_SCORE_THRESHOLD_ENV,
        str(DEFAULT_EVIDENCE_SCORE_THRESHOLD),
    )
    try:
        threshold = float(raw_threshold)
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=f"{EVIDENCE_SCORE_THRESHOLD_ENV} must be a number from -1 to 1.",
        ) from error

    if not -1.0 <= threshold <= 1.0:
        raise HTTPException(
            status_code=500,
            detail=f"{EVIDENCE_SCORE_THRESHOLD_ENV} must be a number from -1 to 1.",
        )
    return threshold


def has_sufficient_evidence(
    retrieved_chunks: list[dict[str, object]], threshold: float
) -> bool:
    """Use the highest-ranked chunk score as a simple relevance check."""
    if not retrieved_chunks:
        return False
    return float(retrieved_chunks[0]["score"]) >= threshold


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    """Check scope and evidence before answer generation is implemented."""
    if is_out_of_scope(request.question):
        return {
            "status": "out_of_scope",
            "answer": (
                "I can help with selected employment topics covered by GOV.UK. "
                "This question is outside the current topic scope."
            ),
            "citations": [],
        }

    threshold = get_evidence_score_threshold()
    retrieved_chunks = retrieve_top_five(
        request.question,
        get_qdrant_client(),
        get_embedding_model(),
        DEFAULT_COLLECTION,
    )
    if not has_sufficient_evidence(retrieved_chunks, threshold):
        return {
            "status": "insufficient_evidence",
            "answer": (
                "I could not find enough relevant information in the selected "
                "GOV.UK sources to answer that question."
            ),
            "citations": [],
        }

    raise HTTPException(
        status_code=501,
        detail="The ask endpoint is not implemented yet.",
    )
