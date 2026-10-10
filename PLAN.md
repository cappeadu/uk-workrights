# Project Plan: Worker Employment Information Assistant

**Suggested GitHub repository name:** `uk-workrights`

Track work by checking off the stages below. Install dependencies as each phase needs them; this plan lists suggestions and does not require installing everything at the start.

## Phase 0 — Project environment and dependencies

**Build:** Prepare a reproducible local development environment and decide how dependencies will be recorded. Suggested starting point: Python 3.13, `uv` for virtual environments and a lockfile, Git, and Docker Desktop with Docker Compose. Check selected package compatibility before pinning Python and dependency versions. Current Sentence Transformers guidance recommends Python 3.10 or newer; the model stack should be smoke-checked on the chosen interpreter before committing to it. [Python releases](https://www.python.org/downloads/), [Sentence Transformers installation](https://sbert.net/docs/installation.html)

**Suggested dependencies, added when needed:**

- Setup and development: `uv`, `pytest`, `ruff`.
- Source ingestion: `httpx`.
- Retrieval: `sentence-transformers` with CPU PyTorch, `qdrant-client`, and a Qdrant container.
- API and answers: `fastapi`, `uvicorn[standard]`, `anthropic`, and optionally `pydantic-settings` for environment-based configuration.
- UI: `streamlit`.

**Stages**

- [ ] Confirm Git, Python, and Docker Compose are available locally.
- [ ] Create a Python virtual environment and choose the dependency/lockfile workflow (recommended: `uv`, `pyproject.toml`, and `uv.lock`).
- [ ] Add the first dependencies only when their phase begins; pin the versions that work together.
- [ ] Keep local credentials in an ignored `.env` file and provide only a safe example template if configuration is added.
- [ ] Check that generated data, vector-store files, model downloads, and caches stay out of Git.

**Success:** The development environment is repeatable from its declared Python and dependency files; the selected Python version can install the CPU embedding stack; Docker Compose can run a local container; and no secret or runtime corpus data is tracked.

## Phase 1 — Source and question pilot

**Build:** Approve the worker-facing topic scope, review an initial allowlist of about 8–12 page-specific GOV.UK sources, and create a small labeled question set. Confirm each source's actual content path, status, and content type from its Content API response. Record supported source pages and facts for 10 worker questions. Do not discover sources by crawling links.

**Stages.**

- [ ] Finalize the worker-facing topics and define out-of-scope handling for questions beyond the selected v1 guidance.
- [ ] Select and review the initial allowlist, including sources for contracts, sick leave, and maternity guidance as well as the other in-scope topics.
- [ ] Verify each selected page through the Content API; record redirects, withdrawn pages, empty responses, or placeholders for a decision.
- [ ] Record each source's provenance and applicable attribution/licence information.
- [ ] Write 10 questions with expected supporting facts and source pages; include at least one ambiguous, one unsupported, and one out-of-scope example.

**Success:** The reviewed source list and question set are approved, each source has a verified path and provenance record, and all unresolved API responses are visible rather than silently omitted.

## Phase 2 — Ingestion and source-linked chunks

**Build:** Add a one-time ingestion command that fetches only the reviewed allowlist, saves inspectable local responses, extracts readable content while retaining headings, lists, and tables, and creates source-linked chunks. Store generated corpus data locally, not in Git.

**Stages**

- [ ] Implement allowlist-driven Content API requests with clear reports for missing, redirected, withdrawn, empty, and failed pages.
- [ ] Extract readable text from API content while preserving meaningful structure.
- [ ] Record title, canonical path, content ID, document/schema type, publishing timestamps, withdrawal status, fetch time, content hash, and attribution.
- [ ] Create chunks with stable IDs and source metadata; verify every chunk resolves to its original source record.
- [ ] Inspect fetched responses and representative chunks for every approved source.

**Success:** One command builds the corpus from the allowlist, every accepted chunk traces to a source and includes required metadata, and ingestion issues are reported for review.

## Phase 3 — Retrieval baseline

**Build:** Embed chunks and queries with the fixed local model `sentence-transformers/all-MiniLM-L6-v2`. Load vectors, text, and metadata into Qdrant. Test retrieval without Claude. Expand the labeled set to 20–30 questions and record Recall@5, measuring the macro-average over questions of the fraction of labeled supporting chunks found in the top five.

**Stages**

- [ ] Add local model loading and chunk embedding using CPU inference as the baseline.
- [ ] Create/load the Qdrant collection with matching vector dimensions and source metadata.
- [ ] Implement top-five retrieval and a repeatable evaluation command.
- [ ] Expand annotations to 20–30 questions with expected supporting chunks/facts.
- [ ] Record Recall@5 and investigate misses before proceeding to answer generation.

**Success:** The corpus loads into Qdrant, retrieval returns source-linked evidence, and evaluation Recall@5 is at least 90% on the labeled set.

## Phase 4 — Grounded FastAPI

**Build:** Implement the answer API over retrieval. Proposed request: `POST /ask` with `question`. Proposed response: `status` (`answer`, `insufficient_evidence`, or `out_of_scope` for topics beyond the v1 guidance), `answer`, and `citations` containing source IDs, titles, excerpts, and URLs rendered by the backend. `GET /health` returns a simple health status.

Claude should only be called for in-scope questions with supporting evidence. It returns answer text and source IDs for backend validation; it must not provide trusted URLs. Invalid citations or no usable evidence produce a clear abstention. Configure the Claude model explicitly and provide its API key through the environment.

**Stages**

- [ ] Add `GET /health` and input validation for `POST /ask`.
- [ ] Handle questions outside the selected v1 guidance topics as clearly out of scope.
- [ ] Add a configurable evidence sufficiency check before calling Claude.
- [ ] Prompt Claude to answer only from supplied evidence and return source IDs.
- [ ] Validate IDs against retrieved records and render all citation details server-side.
- [ ] Test supported, ambiguous, unsupported, insufficient-evidence, invalid-citation, and out-of-scope-topic cases.
- [ ] Confirm questions and conversations are not persisted and secrets are not logged or committed.

**Success:** The API answers supported questions from retrieved evidence, cites only records actually supplied to the model, abstains when evidence is insufficient, and returns a clear out-of-scope response for topics beyond the selected v1 guidance.

## Phase 5 — Streamlit UI and local packaging

**Build:** Add a Streamlit chat page with a question input, answer, and expandable source excerpts/links. Add Dockerfiles and Compose configuration for Streamlit, FastAPI, and Qdrant. Keep this local; cloud hosting is out of scope.

**Stages**

- [ ] Build the chat input, answer display, abstention state, and citation inspection UI.
- [ ] Connect Streamlit to the FastAPI endpoint; do not store accounts or persistent chat history.
- [ ] Add local container configuration for the UI and API plus Qdrant with persistent local vector storage.
- [ ] Document required environment variables and local start steps without including credentials.
- [ ] Run the end-to-end evaluation question set through the local UI and inspect citation links.

**Success:** A developer can start the complete app locally with Docker Compose, ask a question, and inspect source evidence; the retrieval target still passes, all displayed citations resolve to retrieved source records, and unsupported questions clearly abstain.

## Later versions

Later versions can expand the curated GOV.UK source collection to cover topics beyond employment guidance, based on user needs and evaluation results.
