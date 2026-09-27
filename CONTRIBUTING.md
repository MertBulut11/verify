# Contributing

Four people, one repo. This keeps everyone out of each other's way.

## Branches

- `main` is always the working/demoable state. Don't push directly to it.
- Work on a branch named after your module and task:
  `ai-engine/claim-extraction`, `search/reranking`, `backend/api-endpoints`,
  `frontend/results-screen`.
- Open a pull request into `main` when your piece works. At least one
  other teammate reviews before merging — doesn't have to be a deep
  review, just a sanity check that it runs and matches the module
  contract in the root README.

## Commits

Short, present-tense, specific:

```
ai-engine: add source-citation trap test
search: switch reranker to cross-encoder
backend: wire verify_claim into /check endpoint
```

## Before opening a PR

- Your module's own tests pass (each module folder explains how to run
  them).
- You haven't changed another module's files unless you talked to its
  owner first.
- If you changed a function signature or return shape that another
  module depends on (see "Module contracts" in the root README), say
  so explicitly in the PR description and ping that person.

## Secrets

Never commit `.env` or real API keys — `.gitignore` already excludes
`.env`, only `.env.example` (with placeholder values) is tracked. If
you accidentally commit a real key, rotate it immediately, don't just
delete the commit.

## Weekly sync

Since everyone's module depends on the ones before it in the pipeline
(ai-engine → search-retrieval → backend → frontend), it's worth a quick
sync whenever a module's output shape changes, rather than finding out
at integration time.
