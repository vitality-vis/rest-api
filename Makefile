.PHONY: test test-live test-all

PYTEST ?= pytest
PYTEST_ARGS ?=
TESTS ?= tests
API_BASE_URL ?= http://127.0.0.1:3000
export API_BASE_URL

# Local unit tests: no credentials or live Zilliz access required. By default,
# this includes every test in tests/.
test:
	$(PYTEST) $(TESTS) -m "not live" -s $(PYTEST_ARGS)

# Live checks against the configured Zilliz service and an already-running API.
# Set API_BASE_URL to target a different API server.
test-live:
	$(PYTEST) $(TESTS) -m live -s $(PYTEST_ARGS)

# Run all unit and live checks against the configured API server.
test-all:
	$(PYTEST) $(TESTS) $(PYTEST_ARGS)
