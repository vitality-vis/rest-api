.PHONY: test test-live test-all

PYTEST ?= pytest
PYTEST_ARGS ?=
TESTS ?= tests

# Explicitly selected API smoke tests target an already-running local server by
# default. Keep API_BASE_URL unset for the normal live suite, so API tests
# continue to skip when no server was explicitly selected.
ifneq (,$(filter tests/test_api_%.py,$(TESTS)))
API_BASE_URL ?= http://127.0.0.1:3000
export API_BASE_URL
endif

# Local unit tests: no credentials or live Zilliz access required. By default,
# this includes every test in tests/.
test:
	$(PYTEST) $(TESTS) -m "not live" -s $(PYTEST_ARGS)

# Read-only check against the configured live Zilliz service.
test-live:
	$(PYTEST) $(TESTS) -m live -s $(PYTEST_ARGS)

# Run unit checks and live checks against the already-running local API by default.
# Pass API_BASE_URL to target a different server.
test-all:
	API_BASE_URL="$(if $(API_BASE_URL),$(API_BASE_URL),http://127.0.0.1:3000)" $(PYTEST) $(TESTS) $(PYTEST_ARGS)
