# rest-api tests

Run these commands from `rest-api/` using the same Python environment as the API:

```bash
make test       # Local tests: no credentials or external services
make test-live  # Live Zilliz and API smoke tests (API must be running)
make test-all   # Both local and live tests (API must be running)
```

## Test a running API server

API smoke tests send real HTTP requests to a server that is already running.
They do not start a server or mock cache/Zilliz data.

Start the API normally, then test your local server:

```bash
make test-live TESTS=tests/test_api_bootstrap.py
make test-live TESTS=tests/test_api_getpapers.py
make test-live TESTS=tests/test_api_paper_citations.py
```

Make defaults to `http://127.0.0.1:3000`. To test another running server, set
its URL:

```bash
API_BASE_URL=https://example.com make test-live TESTS=tests/test_api_bootstrap.py
```

`make test-live` targets `http://127.0.0.1:3000` by default. Start the API
before running it, or set `API_BASE_URL` to a different running server.

To run one test file:

```bash
make test TESTS=tests/test_static_cache.py
```

The Zilliz live check needs `ZILLIZ_URI` and `ZILLIZ_TOKEN` (for example in
`.env`) and never refreshes local data files.

Full data export and startup-refresh checks are deliberately outside the test
suite; use `python script/export_zilliz_static_data.py` or `python main.py`
when you explicitly need them.
