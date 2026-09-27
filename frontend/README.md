# frontend — Person 4

Owns: web UI, results screen, source display, and eventually a mobile
app / "Share" and "Quick Verify" features.

## What this module needs from the backend

Once `backend/` exposes its API (see `backend/README.md`), the frontend
mainly needs to render something like:

```json
{
  "claims": [
    {
      "id": "claim_1",
      "text": "Apple, Türkiye'de bir fabrika açmıştır.",
      "verdict": "REFUTED",
      "explanation": "Sağlanan kanıtlara göre ... [kaynak_1] ... [kaynak_2].",
      "sources": [
        {"id": "kaynak_1", "url": "https://...", "title": "..."},
        {"id": "kaynak_2", "url": "https://...", "title": "..."}
      ]
    }
  ]
}
```

Exact shape isn't final — confirm it with Person 3 once `backend/` has
a working `/check` endpoint, since the field names above are a
starting proposal, not a contract yet.

## UI pieces worth planning for

- An input box for the user's text/claim.
- A results view: one card per claim, showing the verdict (color-coded:
  e.g. green/SUPPORTED, red/REFUTED, gray/NOT ENOUGH EVIDENCE,
  yellow/CONFLICTING), the explanation text, and clickable citation
  links (`[kaynak_1]` → the matching source's URL).
- A loading/pending state — the pipeline (extract → search → verify)
  takes a few seconds per claim, so the UI shouldn't look frozen.
- Later: a "Quick Verify" flow (paste a link or short text and get a
  fast single-claim check) and a share feature.

## Things worth deciding early

- Framework (React/Vue/plain HTML+JS) and whether this becomes a
  single-page app talking to `backend/`'s API over HTTP.
- How to render inline citations (`[kaynak_1]`) as clickable links
  without breaking if `ai-engine` ever changes the citation format —
  worth confirming that format with Person 1 stays stable.

## Setup

_TODO: add your framework's setup steps (e.g. `npm install`,
`npm run dev`) here once the stack is chosen._
