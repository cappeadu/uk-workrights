"""FastAPI endpoints for the Worker Employment Information Assistant."""

from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

app = FastAPI(title="Worker Employment Information Assistant")


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


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> None:
    """Validate a question until answer generation is implemented."""
    raise HTTPException(
        status_code=501,
        detail="The ask endpoint is not implemented yet.",
    )
