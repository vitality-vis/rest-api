"""Dependency-boundary checks for papers vs full/rerank extras."""

from __future__ import annotations

import ast
from pathlib import Path


REST_API_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_API_ROOT = REST_API_ROOT / "app" / "api" / "public"
RAG_CORE_PATH = REST_API_ROOT / "agents" / "agent_v1_legacy" / "rag_core.py"

FORBIDDEN_TOP_LEVEL = {
    "torch",
    "sentence_transformers",
    "transformers",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langchain_community",
    "flask_socketio",
    "socketio",
    "google",
}


def _imported_top_level_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".", 1)[0])
    return names


def test_public_api_modules_avoid_full_only_top_level_imports():
    for path in sorted(PUBLIC_API_ROOT.rglob("*.py")):
        imported = _imported_top_level_modules(path)
        forbidden = imported & FORBIDDEN_TOP_LEVEL
        assert not forbidden, f"{path.relative_to(REST_API_ROOT)} imports {sorted(forbidden)}"


def test_legacy_rag_core_does_not_eagerly_import_cross_encoder():
    source = RAG_CORE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RAG_CORE_PATH))
    top_level = _imported_top_level_modules(RAG_CORE_PATH)
    assert "sentence_transformers" not in top_level

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CROSS_ENCODER_MODEL":
                    raise AssertionError("legacy CrossEncoder must not be eagerly constructed")


def test_pyproject_declares_papers_full_rerank_dev_extras():
    text = (REST_API_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "vitality-rest-api"' in text
    assert "[project.optional-dependencies]" in text
    assert "\nfull = [" in text
    assert "\nrerank = [" in text
    assert "\ndev = [" in text
    assert '"openai"' in text
    assert "sentence-transformers" in text
    assert "torch" in text
    # Base must not pin the heavy local stack.
    base_section = text.split("[project.optional-dependencies]", 1)[0]
    assert "sentence-transformers" not in base_section
    assert "torch" not in base_section
    assert "langchain" not in base_section


def test_requirements_txt_is_extras_shim():
    lines = [
        line.strip()
        for line in (REST_API_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines == ["-e .[full,rerank]"]
