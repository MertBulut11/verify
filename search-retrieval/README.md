# search-retrieval — Person 2

Owns: web search, scraping, chunking, embeddings, BM25/vector search,
reranking, source selection.

## What this module needs to hand off

`ai-engine`'s `verify_claim()` expects a list of `Source` objects for a
given claim:

```python
class Source:
    id: str     # e.g. "kaynak_1" — referenced in the verdict's citations
    url: str    # link the frontend can show/link to
    text: str   # the actual evidence snippet, ideally a short passage,
                # not a whole page
```

A reasonable contract for this module:

```python
def search(claim_text: str, max_sources: int = 5) -> list[Source]:
    """
    Given an atomic claim, return up to `max_sources` labelled evidence
    snippets, ranked by relevance. Include the source URL so the
    frontend can show it, and keep `text` focused (a paragraph, not a
    full article) so the verdict step doesn't drown in noise.
    """
```

## Things worth deciding early

- Where evidence comes from: a live search API (e.g. a web search API),
  a fixed set of trusted domains, or a pre-built local corpus/database —
  or some mix.
- How you rank/rerank candidates before handing the top few to
  `ai-engine` (BM25, embeddings + cosine similarity, or a
  cross-encoder reranker).
- What "no good evidence found" looks like — return an empty list
  rather than forcing a weak result; `ai-engine`'s verdict step already
  handles `NOT ENOUGH EVIDENCE` when the evidence list doesn't support
  a claim.
- Keep an eye out for near-duplicate sources (same story, multiple
  outlets) — dedupe before sending, so `used_sources` citations don't
  end up redundant.

## Setup

_TODO: add your `requirements.txt` and setup steps here once you've
picked a stack (e.g. `requests`/`beautifulsoup4` for scraping,
`sentence-transformers` for embeddings, a vector store, etc.)._
