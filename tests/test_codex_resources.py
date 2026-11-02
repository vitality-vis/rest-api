"""Versioned Codex instructions and portable skill installation."""

from pathlib import Path

from app.codex.resources import install_managed_skills, load_developer_instructions


def test_instructions_are_loaded_from_versioned_markdown():
    instructions = load_developer_instructions()

    assert "VitaLITy's collaborative research assistant" in instructions
    assert "summarize-selected-papers" in instructions


def test_managed_skill_install_is_portable_and_preserves_other_skills(tmp_path):
    unrelated = tmp_path / "skills" / "other-skill" / "SKILL.md"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("other", encoding="utf-8")

    installed = install_managed_skills(tmp_path)
    target = tmp_path / "skills" / "summarize-selected-papers" / "SKILL.md"

    assert installed == [target.parent]
    assert target.is_file()
    assert not target.parent.is_symlink()
    assert unrelated.read_text(encoding="utf-8") == "other"

    target.write_text("stale", encoding="utf-8")
    install_managed_skills(tmp_path)
    assert target.read_text(encoding="utf-8") != "stale"
