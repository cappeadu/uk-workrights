# Project guidance for coding agents

## Project purpose and scope

Build the GOV.UK Employment Guidance Assistant described in [MVP.md](MVP.md). The phased implementation plan and phase acceptance criteria are in [PLAN.md](PLAN.md); follow them in order and work only on the current stage the user has requested.

- Help workers and employees find and understand employment guidance from a reviewed set of GOV.UK pages.
- Do not ask users for location information or classify sources by location. Present answers as summaries of the cited GOV.UK content, not as a decision about an individual's rights.
- Show evidence and abstain when the evidence is insufficient. Do not present the service as legal advice.
- Keep the corpus curated. Fetch only allowlisted page paths through the GOV.UK Content API; do not crawl related pages automatically.
- Preserve each source's provenance. Validate citation IDs on the backend and render URLs from stored source records, never from model output.
- Keep the first version local. Do not add accounts, persistent chat history, cloud hosting, hybrid search, reranking, agents, or other excluded features without explicit direction.

## Work with the user

- Implement one requested stage at a time, following its build steps and success criteria in `PLAN.md`.
- Do not install dependencies, start services, run ingestion, or make unrelated edits unless the user asks for that work. The user installs dependencies themselves.
- Do not create, edit, move, or delete project files unless the user explicitly asks for the change. In particular, do not automatically update plan checkboxes or add follow-on improvements.
- Before making changes, inspect the relevant files and existing work. Preserve user changes and keep edits limited to the requested stage.
- After implementation, explain what changed, which files changed, and what verification was performed. If the user needs to run a command or program after reviewing the response, provide the exact CLI command and explain its purpose.
- Do not commit or push automatically. When a completed stage is ready for a commit, give the user a concise, clear commit message to use.

## Implementation style

- Prefer the simplest working design that meets the MVP. Write code that is easy to read, run, and debug; avoid speculative abstractions, extra frameworks, and unnecessary dependencies.
- Keep behavior and configuration explicit. Use clear names, small focused functions, useful error messages, and types where they make the code easier to understand.
- Keep API credentials out of source control. Read secrets from environment configuration, and ensure `.env` and generated runtime data remain ignored.
- Keep source text, chunks, and citations traceable to the original page. Surface missing, redirected, withdrawn, or empty API responses for review; never silently skip them.
- Treat all model output as untrusted. The backend decides whether evidence is sufficient, validates source IDs, and supplies citation titles, excerpts, and links from stored records.

## Repository context

- Suggested GitHub repository name: `uk-workrights`.
- Python project name: `workrights-assistant`; import package: `workrights_assistant`.
- Phase 0 uses Python 3.13 and `uv`; dependency declarations belong in `pyproject.toml` and the resolved lockfile in `uv.lock`.
- Add dependencies only when their phase needs them. Current development dependencies are `pytest` and `ruff`; see `PLAN.md` for suggested runtime packages.
- Code layout: `src/workrights_assistant/` for the Python package, `ui/` for Streamlit, `scripts/` for command-line entry points, `config/` for reviewed configuration, and `tests/` for tests and evaluation fixtures.
- `data/raw/`, `data/processed/`, `qdrant_storage/`, model downloads, local caches, virtual environments, and credentials are local runtime artifacts and must not be committed.
- Use FastAPI for `GET /health` and `POST /ask`, Qdrant for local vector storage, `sentence-transformers/all-MiniLM-L6-v2` as the fixed local embedding model, Claude for answer generation when evidence supports a response, and Streamlit for the UI, as scoped in the MVP and plan.

## Verification and commits

- Follow the current phase's checks in `PLAN.md`. Do not run destructive commands or install packages as a side effect of verification.
- Report checks accurately; distinguish checks that were run from commands the user still needs to run.
- Do not commit secrets, downloaded corpora, local vector-store files, model weights, or environment folders.
