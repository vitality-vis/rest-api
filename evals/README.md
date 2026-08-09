# Router evaluation

`router_cases.jsonl` is a hand-labelled regression set for the top-level `agent_v2` router. Each line defines the routing context and expected `route` / `response_mode`; cases may optionally assert a partial `search_intent` (only listed keys are checked). This evaluator does not run paper retrieval or chat answer generation.

Run it from `rest-api/` with the same virtual environment and Azure settings as the API:

```bash
venv/bin/python evals/run_router_eval.py
```

Use `--cases` to evaluate a different JSONL file, `--max-cases` for a cheap smoke sample, and `--output` to choose a result path. Results are written as JSONL, followed by a JSON summary printed to stdout. Default result files are created in `evals/results/` and ignored by Git.

The summary reports route accuracy, response-mode accuracy conditional on an expected `search` route, optional search-intent accuracy for cases that label `expected.search_intent`, joint accuracy (route + mode + intent), expected-search recall, false-search rate, and router decision-status counts. The default is one run per case; use `--repeat 3` or another value when you want to measure decision stability from the non-deterministic deployed chat model.

When changing router policy or prompt, add a representative labelled case first and compare a fresh evaluation result with the saved baseline.
