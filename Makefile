.PHONY: test test-live test-all

PYTEST ?= pytest
PYTEST_ARGS ?=
TESTS ?= tests

# Local unit tests: no credentials or live Zilliz access required. By default,
# this includes every test in tests/.
test:
	$(PYTEST) $(TESTS) -m "not live" -s $(PYTEST_ARGS)

# Read-only check against the configured live Zilliz service.
test-live:
	$(PYTEST) $(TESTS) -m live -s $(PYTEST_ARGS)

# Run both groups. Live checks skip themselves when credentials are unavailable.
test-all:
	$(PYTEST) $(TESTS) $(PYTEST_ARGS)
