"""Authenticated project lifecycle API."""

from __future__ import annotations

from uuid import UUID

from flask import Blueprint, Response, current_app, jsonify, request
from flask_cors import cross_origin

from repositories.supabase.auth import (
    SupabaseAuthenticationError,
    SupabaseConfigurationError,
    verify_access_token,
)
from repositories.supabase.projects_repository import (
    ProjectConflictError,
    ProjectNotFoundError,
    ProjectsPersistenceError,
    ProjectValidationError,
    create_project,
    list_projects,
    rename_project,
)

projects_bp = Blueprint("projects", __name__)
MAX_PROJECT_NAME_LENGTH = 120
PROJECT_RESPONSE_FIELDS = (
    "id", "name", "description", "is_default", "created_at", "updated_at"
)


def _require_auth() -> tuple[tuple[str, str] | None, Response | None]:
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None, Response("Unauthorized", status=401, mimetype="text/plain")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None, Response("Unauthorized", status=401, mimetype="text/plain")
    token = token.strip()
    try:
        return (verify_access_token(token), token), None
    except SupabaseConfigurationError:
        return None, Response("Projects are unavailable", status=503, mimetype="text/plain")
    except SupabaseAuthenticationError:
        return None, Response("Unauthorized", status=401, mimetype="text/plain")


def _project_response(project: dict[str, object]) -> dict[str, object]:
    return {field: project.get(field) for field in PROJECT_RESPONSE_FIELDS}


def _validated_name(payload: object) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        raise ValueError("name is required")
    name = payload["name"].strip()
    if not name:
        raise ValueError("name is required")
    if len(name) > MAX_PROJECT_NAME_LENGTH:
        raise ValueError(f"name must be at most {MAX_PROJECT_NAME_LENGTH} characters")
    return name


def _project_id(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as error:
        raise ValueError("invalid project id") from error


def _mutation_error(error: ProjectsPersistenceError):
    if isinstance(error, ProjectValidationError):
        return jsonify({"error": str(error)}), 400
    if isinstance(error, ProjectNotFoundError):
        return jsonify({"error": "Project not found"}), 404
    if isinstance(error, ProjectConflictError):
        return jsonify({"error": str(error)}), 409
    current_app.logger.error("Project operation failed: %s", error)
    return Response("Projects are unavailable", status=503, mimetype="text/plain")


@projects_bp.route("/projects", methods=["GET"])
@cross_origin()
def get_projects():
    auth, error = _require_auth()
    if error is not None:
        return error
    try:
        projects = list_projects(user_id=auth[0], access_token=auth[1])
    except ProjectsPersistenceError as persistence_error:
        current_app.logger.error("Could not load projects: %s", persistence_error)
        return Response("Projects are unavailable", status=503, mimetype="text/plain")
    return jsonify({"projects": [_project_response(project) for project in projects]})


@projects_bp.route("/projects", methods=["POST"])
@cross_origin()
def post_project():
    auth, error = _require_auth()
    if error is not None:
        return error
    payload = request.get_json(silent=True)
    try:
        name = _validated_name(payload)
        assert isinstance(payload, dict)
        if (
            "description" in payload
            and payload["description"] is not None
            and not isinstance(payload["description"], str)
        ):
            raise ValueError("description must be a string or null")
        project = create_project(
            access_token=auth[1], name=name, description=payload.get("description")
        )
    except ValueError as validation_error:
        return jsonify({"error": str(validation_error)}), 400
    except ProjectsPersistenceError as persistence_error:
        return _mutation_error(persistence_error)
    return jsonify(_project_response(project)), 201


@projects_bp.route("/projects/<project_id>", methods=["PATCH"])
@cross_origin()
def patch_project(project_id: str):
    auth, error = _require_auth()
    if error is not None:
        return error
    payload = request.get_json(silent=True)
    try:
        normalized_id = _project_id(project_id)
        name = _validated_name(payload)
        assert isinstance(payload, dict)
        if set(payload) != {"name"}:
            raise ValueError("only name may be changed")
        project = rename_project(access_token=auth[1], project_id=normalized_id, name=name)
    except ValueError as validation_error:
        return jsonify({"error": str(validation_error)}), 400
    except ProjectsPersistenceError as persistence_error:
        return _mutation_error(persistence_error)
    return jsonify(_project_response(project))
