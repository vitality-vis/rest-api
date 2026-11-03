"""Focused tests for the project shelf and user-level asset boundary."""
from unittest.mock import patch

import pytest
from flask import Flask

from app.api.user.library import library_bp

PROJECT = {"id": "project-1", "user_id": "user-1"}
ASSET = {"id": "asset-1", "paper_id": "paper-1", "origin": "user", "metadata_snapshot": {"ID": "paper-1", "Title": "T"}, "azure_file_id": "file-1"}

@pytest.fixture
def client():
    app = Flask(__name__); app.register_blueprint(library_bp); return app.test_client()

def auth(): return {"Authorization": "Bearer token"}

def import_item():
    return {"paper": {"ID": "user:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "title": "Imported", "abstract": "Abstract"}}

def test_unscoped_shelf_mutation_rules_are_absent(client):
    rules = {(rule.rule, tuple(sorted(rule.methods - {"HEAD", "OPTIONS"}))) for rule in client.application.url_map.iter_rules()}
    removed = {
        ("/library/papers/<path:paper_id>/saved", ("PUT",)),
        ("/library/papers/<path:paper_id>/saved", ("DELETE",)),
        ("/library/papers/saved", ("POST",)),
        ("/library/papers/unsave", ("POST",)),
        ("/library/papers/import", ("POST",)),
        ("/library/papers/<path:paper_id>/tags", ("PUT",)),
    }
    assert rules.isdisjoint(removed)

def test_scoped_unsave_removes_membership_not_asset(client):
    with patch("app.api.user.library.verify_access_token", return_value="user-1"), patch("app.api.user.library.get_project", return_value=PROJECT), patch("app.api.user.library.get_user_paper", return_value=ASSET), patch("app.api.user.library.remove_project_paper", return_value=True) as remove, patch("app.api.user.library.delete_user_paper") as delete:
        response = client.delete("/projects/project-1/library/papers/paper-1/saved", headers=auth())
    assert response.status_code == 204
    remove.assert_called_once_with(user_id="user-1", project_id="project-1", user_paper_id="asset-1")
    delete.assert_not_called()

def test_scoped_tags_only_update_membership(client):
    with patch("app.api.user.library.verify_access_token", return_value="user-1"), patch("app.api.user.library.get_project", return_value=PROJECT), patch("app.api.user.library.get_user_paper", return_value=ASSET), patch("app.api.user.library.update_project_paper_tags", return_value={"tags": ["x"]}) as membership:
        response = client.put("/projects/project-1/library/papers/paper-1/tags", headers=auth(), json={"tags": ["x"]})
    assert response.status_code == 200
    membership.assert_called_once_with(user_id="user-1", project_id="project-1", user_paper_id="asset-1", tags=["x"])

def test_permanent_delete_guard_precedes_azure_side_effects(client):
    with patch("app.api.user.library.verify_access_token", return_value="user-1"), patch("app.api.user.library.get_user_paper", return_value=ASSET), patch("app.api.user.library.has_any_membership", return_value=True), patch("app.api.user.library.detach_user_paper_file") as detach, patch("app.api.user.library.delete_azure_file") as azure:
        response = client.delete("/library/papers/paper-1", headers=auth())
    assert response.status_code == 409
    detach.assert_not_called(); azure.assert_not_called()

def test_scoped_import_checks_project_and_adds_membership(client):
    persisted = [{**ASSET, "paper_id": import_item()["paper"]["ID"]}]
    with patch("app.api.user.library.verify_access_token", return_value="user-1"), patch("app.api.user.library.get_project", return_value=PROJECT) as project, patch("app.api.user.library.import_user_papers", return_value=persisted) as imported, patch("app.api.user.library.add_project_paper") as add:
        response = client.post("/projects/project-1/library/papers/import", headers=auth(), json={"items": [import_item()], "also_save": True})
    assert response.status_code == 200
    project.assert_called_once_with(user_id="user-1", project_id="project-1")
    imported.assert_called_once()
    assert "is_saved" not in imported.call_args.kwargs
    add.assert_called_once_with(user_id="user-1", project_id="project-1", user_paper_id="asset-1")

def test_import_without_membership_still_requires_owned_project(client):
    with patch("app.api.user.library.verify_access_token", return_value="user-1"), patch("app.api.user.library.get_project", return_value=PROJECT) as project, patch("app.api.user.library.import_user_papers", return_value=[ASSET]), patch("app.api.user.library.add_project_paper") as add:
        response = client.post("/projects/project-1/library/papers/import", headers=auth(), json={"items": [import_item()], "also_save": False})
    assert response.status_code == 200
    project.assert_called_once(); add.assert_not_called()
