# rest-api tests

Run these commands from `rest-api/` using the same Python environment as the API:

```bash
make test       # Local tests: no credentials or external services
make test-live  # Read-only checks against real external services
make test-all   # Both; live checks skip when credentials are unavailable
```

## Test a running API server

API smoke tests send real HTTP requests to a server that is already running.
They do not start a server or mock cache/Zilliz data.

Start the API normally, then test your local server:

```bash
make test-live TESTS=tests/test_api_bootstrap.py
make test-live TESTS=tests/test_api_getpapers.py
```

When an explicitly selected test matches `tests/test_api_*.py`, Make defaults
to `http://127.0.0.1:3000`. To test another running server, set its URL:

```bash
API_BASE_URL=https://example.com make test-live TESTS=tests/test_api_bootstrap.py
```

The normal `make test-live` command does not select an API server, so API smoke
tests skip themselves in that suite.

To run one test file:

```bash
make test TESTS=tests/test_static_cache.py
```

The Zilliz live check needs `ZILLIZ_URI` and `ZILLIZ_TOKEN` (for example in
`.env`) and never refreshes local data files.

Full data export and startup-refresh checks are deliberately outside the test
suite; use `python script/export_zilliz_static_data.py` or `python main.py`
when you explicitly need them.
