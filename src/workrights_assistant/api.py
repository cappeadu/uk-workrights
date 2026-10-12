"""FastAPI endpoints for the Worker Employment Information Assistant."""

import re
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

app = FastAPI(title="Worker Employment Information Assistant")

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


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    """Flag clear out-of-scope questions until answering is implemented."""
    if is_out_of_scope(request.question):
        return {
            "status": "out_of_scope",
            "answer": (
                "I can help with selected employment topics covered by GOV.UK. "
                "This question is outside the current topic scope."
            ),
            "citations": [],
        }

    raise HTTPException(
        status_code=501,
        detail="The ask endpoint is not implemented yet.",
    )
