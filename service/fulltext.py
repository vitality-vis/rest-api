"""Synchronous, user-retryable library file indexing for the MVP."""

from __future__ import annotations

from repositories.azure_openai.vector_stores import (
    AzureVectorStoresError, attach_file, create_vector_store, detach_file, poll_file_until_terminal,
)
from repositories.supabase.user_papers_repository import (
    get_user_paper, update_user_paper_index_state,
)
from repositories.supabase.user_vector_stores_repository import create_user_vector_store, get_user_vector_store


class LibraryIndexError(RuntimeError):
    pass


INDEX_POLL_ATTEMPTS = 15
INDEX_POLL_INTERVAL_SECONDS = 2


def index_user_paper(*, user_id: str, paper_id: str) -> dict[str, object]:
    paper = get_user_paper(user_id=user_id, paper_id=paper_id)
    if paper is None or not isinstance(paper.get("azure_file_id"), str):
        raise LibraryIndexError("No uploaded PDF is available")
    file_id = paper["azure_file_id"]
    try:
        store = get_user_vector_store(user_id=user_id)
        if store is None:
            created = create_vector_store(name=f"library-{user_id}")
            store_id = created.get("id")
            if not isinstance(store_id, str) or not store_id:
                raise LibraryIndexError("Azure did not return a Vector Store id")
            store = create_user_vector_store(user_id=user_id, azure_vector_store_id=store_id)
        store_id = store.get("azure_vector_store_id")
        if not isinstance(store_id, str):
            raise LibraryIndexError("User Vector Store is invalid")
        attached = attach_file(vector_store_id=store_id, file_id=file_id, attributes={"paper_id": paper_id})
        vs_file_id = attached.get("id")
        if not isinstance(vs_file_id, str) or not vs_file_id:
            raise LibraryIndexError("Azure did not return a Vector Store file id")
        update_user_paper_index_state(user_id=user_id, paper_id=paper_id, azure_file_id=file_id,
                                      status="in_progress", vs_file_id=vs_file_id)
        result = poll_file_until_terminal(vector_store_id=store_id, vector_store_file_id=vs_file_id,
                                          max_attempts=INDEX_POLL_ATTEMPTS,
                                          interval_seconds=INDEX_POLL_INTERVAL_SECONDS)
        status = "completed" if result.get("status") == "completed" else "failed"
        updated = update_user_paper_index_state(user_id=user_id, paper_id=paper_id,
                                                azure_file_id=file_id, status=status,
                                                vs_file_id=vs_file_id,
                                                error=None if status == "completed" else "Azure indexing failed")
        return updated or paper
    except (AzureVectorStoresError, LibraryIndexError) as error:
        update_user_paper_index_state(user_id=user_id, paper_id=paper_id, azure_file_id=file_id,
                                      status="failed", error="Indexing failed; please retry")
        raise LibraryIndexError("Indexing failed; please retry") from error


def detach_user_paper_file(*, user_id: str, paper: dict[str, object]) -> None:
    vs_file_id = paper.get("vs_file_id")
    if not isinstance(vs_file_id, str) or not vs_file_id:
        return
    store = get_user_vector_store(user_id=user_id)
    if store is None or not isinstance(store.get("azure_vector_store_id"), str):
        return
    detach_file(vector_store_id=store["azure_vector_store_id"], vector_store_file_id=vs_file_id)
