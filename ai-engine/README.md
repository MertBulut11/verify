# Fact-Checking System — Person 1 Module (AI / LLM / Claim Verification)

This is the AI "brain" module for the team's fact-checking system. It owns
the two steps that sit between what the user types and what the backend
serves back: turning free text into atomic claims, and turning a claim +
evidence into a verdict.

## Team split (context)

| Person | Responsibility |
|---|---|
| 1 (this module) | Claim extraction, claim decomposition, prompt design, verdict generation, explanation |
| 2 | Web search / scraping, chunking, embeddings, BM25/vector search, reranking, source selection |
| 3 | FastAPI, endpoints, wiring modules together, database/cache, error handling, deployment |
| 4 | Web UI, results screen, source display, later: mobile app |

## What this module does

1. **Claim Extraction** — `extract_claims(text)` splits raw user input
   into atomic, independently verifiable claims. It strips out opinions,
   predictions, and vague statements, resolves pronouns to their real
   subject, and keeps reported speech attributed to whoever said it
   (so Person 2 knows what to actually search for).

2. **Verdict Generation** — `verify_claim(claim_text, sources)` takes a
   claim plus a list of labelled evidence snippets (from Person 2) and
   returns one of `SUPPORTED`, `REFUTED`, `NOT ENOUGH EVIDENCE`, or
   `CONFLICTING`, with an explanation that cites which source ID backed
   each part of the reasoning — and ignores irrelevant/trap sources.

Both calls use Groq's **strict** structured-output mode
(`response_format={"type": "json_schema", ..., "strict": True}`), built
straight from the Pydantic models in this file — so the response is
*guaranteed* to match the schema, no free-text parsing on Backend's side.

## Provider & model choice

**Groq** (https://console.groq.com), using the official `groq` Python
package — an OpenAI-compatible Chat Completions API known for very fast
inference. Default model: `openai/gpt-oss-20b` at `temperature=0.0`.
Swap it via the `FACTCHECK_MODEL_NAME` env var — `openai/gpt-oss-120b`
is the higher-quality option if `20b` isn't accurate enough; both
support Groq's strict JSON-schema mode (`qwen/qwen3.8-27b` also does,
check [Groq's structured-outputs docs](https://console.groq.com/docs/structured-outputs)
for the current list before picking something else, since not every
model on Groq supports `strict: true`).

> This module previously used Gemini (`google-genai`). The team
> switched to Groq — if you're looking at old code/discussion that
> mentions Gemini, it's stale; this file is the source of truth.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then fill in GROQ_API_KEY
```

## Running it

```bash
python ai_engine.py         # quick smoke test (2 calls)
python test_ai_engine.py    # the full edge-case suite below
```

## Edge cases this has been tested against

These were worked out and validated against the model before being
wired into code, and are re-run in `test_ai_engine.py`:

- Nested claims in one sentence (e.g. inflation + minimum wage figures)
- Opinion mixed with fact ("bence..." filtered out)
- Implicit/pronoun references ("o adam...")
- Reported speech, where the *statement itself* is the claim (WHO
  quote, news-site claim) rather than the underlying fact
- Numeric-heavy sentences that should split into multiple claims
  without dropping any
- Fully unverifiable text (predictions, opinions, vague statements) →
  should extract nothing, or flag `is_verifiable: false`
- A "trap" source that's topically unrelated to the claim (must be
  ignored and excluded from `used_sources`)
- A partly-true / partly-false compound claim (must not come back as a
  clean `SUPPORTED` just because the first half checks out)

## Integration contract for Person 3 (Backend)

```python
from ai_engine import FactCheckEngine, Source

engine = FactCheckEngine()  # reads GROQ_API_KEY from env

extraction = engine.extract_claims(user_text)
# -> ExtractionResult(claims=[Claim(id=..., text=..., is_verifiable=...), ...])

# For each verifiable claim, after Person 2's search module returns evidence:
verdict = engine.verify_claim(
    claim_text=claim.text,
    sources=[Source(id="kaynak_1", url="...", text="..."), ...],
)
# -> VerdictResult(claim_id=..., verdict=..., explanation=..., used_sources=[...])
```

Both return Pydantic models — call `.model_dump()` / `.model_dump_json()`
to hand them to FastAPI as-is.

## Suggested next step

Wrap `FactCheckEngine` in a small FastAPI service (or a couple of
endpoints inside the shared API) so Person 3 can call it over HTTP
instead of importing it directly — useful once this needs its own
deploy/scaling story separate from the rest of the backend.
