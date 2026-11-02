"""Versioned Codex instructions and portable skill installation."""

from __future__ import annotations

import shutil
from pathlib import Path

import config

AGENT_SOURCE_DIR = Path(__file__).with_name("agent")
AGENT_INSTRUCTIONS_PATH = AGENT_SOURCE_DIR / "AGENTS.md"
SKILLS_SOURCE_DIR = AGENT_SOURCE_DIR / "skills"
MANAGED_SKILLS = ("summarize-selected-papers",)


def load_developer_instructions() -> str:
    """Load the versioned runtime instructions shipped with the API."""
    return AGENT_INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()


def install_managed_skills(home: str | Path | None = None) -> list[Path]:
    """Copy VitaLITy-owned skills into the active Codex home.

    Copying is intentional: it works when symlinks are unavailable or stripped
    by a deployment system. Only named VitaLITy skills are replaced; unrelated
    skills and all other Codex home state are left untouched.
    """
    target_root = Path(home or config.CODEX_HOME) / "skills"
    installed: list[Path] = []
    for name in MANAGED_SKILLS:
        source = SKILLS_SOURCE_DIR / name
        if not (source / "SKILL.md").is_file():
            raise FileNotFoundError(f"Codex skill source is missing: {source}")
        target = target_root / name
        target_root.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        installed.append(target)
    return installed
