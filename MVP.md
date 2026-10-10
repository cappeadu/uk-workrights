# MVP Proposal: GOV.UK Employment Guidance Assistant

**Suggested GitHub repository name:** `uk-workrights`

## Goal

Build a local assistant that helps workers find and understand a curated set of GOV.UK employment guidance. It must retrieve relevant passages, show the supporting sources, and say when the available evidence is insufficient. The product makes official guidance easier to search; it does not decide how a rule applies to an individual's circumstances.

## Audience and boundaries

- **Audience:** workers and employees asking about their own workplace rights.
- **Topics:** employment status; contracts; holiday entitlement and pay; sick leave and Statutory Sick Pay; working hours; redundancy; dismissal; maternity leave and pay.
- **Product positioning:** users can search the selected guidance without providing location information. Answers are presented as summaries of the linked GOV.UK source.
- **Evidence boundary:** show the source and relevant passages, summarize only what the retrieved evidence supports, and abstain when it is insufficient. Do not present the service as legal advice or decide individual cases.
- **Excluded from v1:** employer payroll or hiring workflows, pensions, job-search advice, PDFs unless a selected source requires one, hybrid search, reranking, agents, accounts, saved chat history, and cloud deployment.

## Source collection

- Maintain a reviewed allowlist of about 8–12 page-specific GOV.UK sources. Candidate topics include employment status, employment contracts, holiday entitlement, sick leave and Statutory Sick Pay, working hours, redundancy, dismissal, and maternity leave and pay. Confirm the exact public page paths and API responses before accepting a source.
- Fetch each page through the GOV.UK Content API using the page's actual path. Use the returned content and metadata, not broad category pages. The API is in beta and may return redirects, placeholders, or empty content, so ingestion must report these for review instead of silently skipping them. [Content API documentation](https://content-api.publishing.service.gov.uk/)
- Preserve title, canonical page path, content ID, document/schema type, publication and update timestamps, withdrawal status, local fetch time, content hash, and applicable attribution.
- Do not crawl related links automatically. Add each source deliberately. Keep attribution and applicable Open Government Licence terms with the corpus and displayed sources. [Open Government Licence guidance](https://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/open-government-licence/)

## Local architecture

1. A one-time ingestion command fetches allowlisted pages, extracts readable text while retaining headings, lists, and tables, and creates source-linked chunks.
2. A fixed local embedding model embeds chunks and questions. Qdrant stores vectors, text, and metadata.
3. FastAPI exposes `GET /health` and `POST /ask`. The ask flow retrieves the top five candidate chunks, checks whether the evidence is sufficient, and calls Claude only when appropriate.
4. Claude answers from supplied evidence and returns source IDs. The backend validates those IDs and renders titles, excerpts, and official links from stored records; the model does not supply trusted URLs.
5. A Streamlit chat UI lets the user search the selected guidance and inspect cited evidence. No location selector, accounts, or persistent chat history.

Keep the embedding model and Claude model explicit in configuration. Keep API credentials out of the repository.

## Build order and completion checks

1. **Source and question pilot:** approve the topics, verify initial source paths, and write 10 questions with expected source pages and supporting facts.
2. **Ingestion:** fetch and inspect real API responses and extracted chunks. Every chunk must trace to an original source, and ingestion problems must be visible for review.
3. **Retrieval:** load Qdrant and evaluate retrieval before Claude is added. Expand the set to about 20–30 questions and record Recall@5.
4. **Grounded API:** add Claude, source-ID validation, and insufficient-evidence responses. Test ordinary, ambiguous, out-of-scope-topic, and unsupported questions.
5. **UI and packaging:** add the chat page and Docker Compose setup for the API, UI, and Qdrant.

The MVP is ready when it runs locally end to end, the evaluation questions retrieve expected evidence reliably, every displayed citation resolves to a source actually used, and unsupported questions produce a clear abstention rather than a guess.
