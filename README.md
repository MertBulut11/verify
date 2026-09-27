# Fact-Checking System

A system that takes a piece of text, breaks it into atomic factual
claims, searches for evidence on each one, and returns a verdict
(`SUPPORTED` / `REFUTED` / `NOT ENOUGH EVIDENCE` / `CONFLICTING`) with
cited sources.

## How it works

```
User text
   │
   ▼
[ai-engine]  extract_claims(text) ──► list of atomic claims
   │
   ▼
[search-retrieval]  search(claim) ──► labelled evidence snippets
   │
   ▼
[ai-engine]  verify_claim(claim, evidence) ──► verdict + citations
   │
   ▼
[backend]  orchestrates the above, exposes it over an API
   │
   ▼
[frontend]  shows the user their claims, verdicts, and sources
```

## Team & modules

| Folder | Owner | Responsibility |
|---|---|---|
| [`ai-engine/`](./ai-engine) | Person 1 | Claim extraction, claim decomposition, prompt design, verdict generation, explanations |
| [`search-retrieval/`](./search-retrieval) | Person 2 | Web search/scraping, chunking, embeddings, BM25/vector search, reranking, source selection |
| [`backend/`](./backend) | Person 3 | API endpoints, wiring the modules together, database/cache, error handling, deployment |
| [`frontend/`](./frontend) | Person 4 | Web UI, results screen, source display, later a mobile app |

Each folder is that person's workspace: its own README, its own
dependencies, and its own tests. `backend/` is what ties everyone's
work together into one running system.

## Module contracts

These are the shapes each module hands to the next one. Keep to these
so nobody's module breaks when another one changes internally.

**ai-engine → everyone:**
```python
extract_claims(text: str) -> ExtractionResult
# ExtractionResult.claims: list[{id, text, is_verifiable}]

verify_claim(claim_text: str, sources: list[Source]) -> VerdictResult
# Source: {id, url, text}
# VerdictResult: {claim_id, verdict, explanation, used_sources}
```

See [`ai-engine/README.md`](./ai-engine/README.md) for the full details,
validated prompts, and edge cases it's been tested against.

**search-retrieval → ai-engine:** for a given claim, return a list of
`Source` objects (`id`, `url`, `text`) — labelled evidence snippets the
verdict step can cite by ID. See
[`search-retrieval/README.md`](./search-retrieval/README.md).

**backend:** owns the end-to-end request flow and the API contract the
frontend calls. See [`backend/README.md`](./backend/README.md).

**frontend:** consumes the backend's API. See
[`frontend/README.md`](./frontend/README.md).

## Getting started (each module)

Every module folder has its own `README.md` and (where applicable)
`requirements.txt` — `cd` into the one you own and follow its setup
instructions. Don't install everything from the repo root; each part
is meant to run and be tested independently until `backend/` wires
them together.

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for branch naming, commit
style, and the PR process.
