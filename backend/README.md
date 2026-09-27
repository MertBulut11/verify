# backend — Person 3

Owns: API endpoints, wiring `ai-engine` and `search-retrieval` together,
database/cache, error handling, deployment.

## What this module does

Sits in the middle of the pipeline and is the only thing the frontend
talks to:

```
1. Receive user text                         (from frontend)
2. ai_engine.extract_claims(text)             -> claims
3. For each verifiable claim:
     search_retrieval.search(claim.text)      -> sources
     ai_engine.verify_claim(claim.text, sources) -> verdict
4. Return claims + verdicts + sources          (to frontend)
```

## Suggested stack

FastAPI, given that's what both the `ai-engine` design chat and this
repo's tooling assume. A minimal shape to start from:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CheckRequest(BaseModel):
    text: str

@app.post("/check")
def check(req: CheckRequest):
    # 1. extract_claims
    # 2. for each claim: search -> verify_claim
    # 3. return combined result
    ...
```

## Integrating the other modules

Once `ai-engine` and `search-retrieval` are stable, import them
directly (or wrap each as its own microservice later if you need to
scale/deploy them separately):

```python
from ai_engine import FactCheckEngine, Source
# from search_retrieval import search   # once Person 2's module exists

engine = FactCheckEngine()
```

## Things worth deciding early

- Sync vs. async: claim extraction, search, and verdict generation are
  all network calls — consider running independent claims concurrently
  (`asyncio.gather`) rather than in a loop, once things work correctly
  in sequence first.
- Caching: repeated claims (e.g. "the earth is round") shouldn't hit
  the LLM or search API every time — a simple cache keyed on claim text
  goes a long way.
- Error handling: what happens if `search-retrieval` returns nothing,
  or `ai-engine` returns a malformed response? Don't let one bad claim
  fail the whole request.
- API shape: agree on the request/response JSON with Person 4
  (frontend) early — see `frontend/README.md`.

## Setup

_TODO: add `requirements.txt` (fastapi, uvicorn, plus whatever
`ai-engine`/`search-retrieval` need) and run instructions here._
