"""Public blueprint import isolation for the papers profile boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from flask import Flask

from app.api.route_allowlist import (
    FULL_BLUEPRINTS,
    FULL_ONLY_BLUEPRINTS,
    PAPERS_BLUEPRINTS,
    load_papers_blueprints,
)


REST_API_ROOT = Path(__file__).resolve().parents[1]


def test_papers_allowlist_excludes_full_only_modules():
    papers_modules = {module for module, _ in PAPERS_BLUEPRINTS}
    full_only_modules = {module for module, _ in FULL_ONLY_BLUEPRINTS}
    assert papers_modules.isdisjoint(full_only_modules)
    assert ("app.api.public.papers", "papers_bp") in PAPERS_BLUEPRINTS
    assert ("app.api.user.papers", "user_papers_bp") in FULL_ONLY_BLUEPRINTS
    assert len(FULL_BLUEPRINTS) == len(PAPERS_BLUEPRINTS) + len(FULL_ONLY_BLUEPRINTS)


def test_load_papers_blueprints_registers_expected_paths():
    app = Flask(__name__)
    for blueprint in load_papers_blueprints():
        app.register_blueprint(blueprint)

    rules = {(rule.rule, tuple(sorted(rule.methods - {"HEAD", "OPTIONS"}))) for rule in app.url_map.iter_rules()}
    assert ("/health", ("GET",)) in rules
    assert ("/getPapers", ("GET", "POST")) in rules
    assert ("/getSimilarPapers", ("POST",)) in rules
    assert ("/getPaperCitations", ("POST",)) in rules
    assert ("/getPaperById", ("GET",)) in rules
    assert ("/getPaperByTitle", ("POST",)) in rules
    assert ("/getUmapPoints", ("GET",)) in rules
    assert ("/getMetaData", ("GET",)) in rules
    assert ("app.api.public.health", "health_bp") in PAPERS_BLUEPRINTS
    assert not any(rule == "/papers/resolve" for rule, _ in rules)
    assert not any(rule == "/getPublicConfig" for rule, _ in rules)
    assert not any(rule == "/checkoutPapers" for rule, _ in rules)


def test_importing_papers_blueprints_skips_supabase_and_agents():
    """Fresh interpreter: public allowlist must not pull supabase or agents."""
    script = """
import sys
from app.api.route_allowlist import load_papers_blueprints

load_papers_blueprints()
forbidden = [
    name
    for name in sys.modules
    if name == "repositories.supabase"
    or name.startswith("repositories.supabase.")
    or name == "agents"
    or name.startswith("agents.")
]
if forbidden:
    raise SystemExit("forbidden imports: " + ", ".join(sorted(forbidden)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REST_API_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "public blueprint import isolation failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
