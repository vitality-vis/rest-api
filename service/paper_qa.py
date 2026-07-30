"""Selected-paper synthesis evidence planning and Azure File Search invocation."""
from __future__ import annotations

from agents.agent_v2.logging import SearchV2Trace
from agents.agent_v2.models import SynthesisExecutionPlan
from repositories.supabase.user_papers_repository import get_user_paper
from repositories.supabase.user_vector_stores_repository import get_user_vector_store
from repositories.azure_openai.vector_stores import (
    AzureVectorStoresError,
    create_file_search_response,
    create_text_response,
    response_output_text,
)
from repositories.zilliz.mappers import paper_to_api_response
from repositories.zilliz.paper_repository import (
    RepositoryUnavailableError,
    get_papers_by_ids,
)


class PaperQAError(RuntimeError):
    pass


def build_evidence_plan(*, user_id: str, paper_ids: list[str]) -> tuple[SynthesisExecutionPlan, str]:
    if not paper_ids:
        raise PaperQAError("Select at least one paper first.")

    searchable: list[str] = []
    for paper_id in paper_ids:
        paper = get_user_paper(user_id=user_id, paper_id=paper_id)
        if paper is None:
            raise PaperQAError("One or more selected papers are unavailable.")
        if paper.get("vs_file_status") == "completed":
            searchable.append(paper_id)

    try:
        catalog_records = get_papers_by_ids(paper_ids)
    except RepositoryUnavailableError as error:
        raise PaperQAError("Selected paper metadata is temporarily unavailable.") from error
    catalog_by_id = {
        str(record.get("paper_uid")): paper_to_api_response(record)
        for record in catalog_records
        if record.get("paper_uid") is not None
    }
    missing_ids = [paper_id for paper_id in paper_ids if paper_id not in catalog_by_id]
    if missing_ids:
        raise PaperQAError("One or more selected papers are unavailable in the paper catalog.")

    metadata: list[str] = []
    for paper_id in paper_ids:
        paper = catalog_by_id[paper_id]
        metadata.append(
            "\n".join(
                [
                    f"Paper ID: {paper_id}",
                    f"Title: {paper.get('Title') or ''}",
                    f"Abstract: {paper.get('Abstract') or ''}",
                    f"Authors: {', '.join(paper.get('Authors') or [])}",
                    f"Keywords: {', '.join(paper.get('Keywords') or [])}",
                    f"Source: {paper.get('Source') or ''}",
                    f"Year: {paper.get('Year') if paper.get('Year') is not None else ''}",
                    f"Citation count: {paper.get('CitationCounts') if paper.get('CitationCounts') is not None else ''}",
                    f"DOI: {paper.get('doi') or ''}",
                ]
            )
        )

    plan = SynthesisExecutionPlan(
        use_file_search=bool(searchable),
        metadata_paper_ids=paper_ids,
        file_search_paper_ids=searchable,
    )
    return plan, "\n\n".join(metadata)


def answer(*, user_id: str, paper_ids: list[str], text: str, trace: SearchV2Trace | None = None) -> str:
    plan, metadata = build_evidence_plan(user_id=user_id, paper_ids=paper_ids)
    if trace:
        trace.log_synthesis_evidence_plan(
            metadata_paper_ids=plan.metadata_paper_ids,
            file_search_paper_ids=plan.file_search_paper_ids,
            use_file_search=plan.use_file_search,
        )
    if not plan.use_file_search:
        input_text = f"Answer using only this selected-paper metadata. Be explicit about uncertainty.\n\nQuestion: {text}\n\n{metadata}"
        if trace:
            trace.log_synthesis_payload(mode="metadata", input_text=input_text)
        try:
            response = create_text_response(input_text=input_text)
        except AzureVectorStoresError as error:
            raise PaperQAError("Metadata synthesis failed. Please try again.") from error
        output = response_output_text(response)
        if output:
            return output
        raise PaperQAError("Metadata synthesis returned no answer.")
    store = get_user_vector_store(user_id=user_id)
    if store is None or not isinstance(store.get("azure_vector_store_id"), str):
        raise PaperQAError("Selected full texts are not ready. Please retry after uploading them.")
    filters = {"type": "in", "key": "paper_id", "value": plan.file_search_paper_ids}
    input_text = f"Use the selected papers to answer:\n{text}\n\nMetadata for all selected papers:\n{metadata}"
    if trace:
        trace.log_synthesis_payload(
            mode="file_search",
            input_text=input_text,
            vector_store_id=store["azure_vector_store_id"],
            filters=filters,
        )
    try:
        response = create_file_search_response(
            input_text=input_text,
            vector_store_id=store["azure_vector_store_id"], filters=filters,
        )
    except AzureVectorStoresError as error:
        raise PaperQAError("Full-text search failed. Please try again.") from error
    output = response_output_text(response)
    if output:
        return output
    raise PaperQAError("Full-text search returned no answer.")
