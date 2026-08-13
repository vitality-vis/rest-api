"""Selected-paper synthesis evidence planning and Azure File Search invocation."""
from __future__ import annotations

import json

from agents.agent_v2.logging import SearchV2Trace
from agents.agent_v2.models import SynthesisExecutionPlan
from repositories.supabase.user_vector_stores_repository import get_user_vector_store
from repositories.azure_openai.vector_stores import (
    AzureVectorStoresError,
    create_file_search_response,
    create_text_response,
    get_azure_vector_stores_settings,
    response_file_citation_annotations,
    response_output_text,
)
from service.grounding import (
    apply_file_citations,
    numbered_paper_ids,
    replace_numbered_citations,
    resolve_file_annotations,
)
from service.paper_registry import (
    LibraryPaperResolutionError,
    resolve_selected_papers,
)


class PaperQAError(RuntimeError):
    pass


SCOPE_WARNING_START = "[[VITALITY_FILE_SEARCH_SCOPE_WARNING]]"
SCOPE_WARNING_END = "[[/VITALITY_FILE_SEARCH_SCOPE_WARNING]]"


def build_evidence_plan(
    *, user_id: str | None, paper_ids: list[str], use_file_search: bool
) -> tuple[SynthesisExecutionPlan, list[str]]:
    if not paper_ids:
        raise PaperQAError("Select at least one paper first.")

    try:
        resolved_papers = resolve_selected_papers(user_id=user_id, paper_ids=paper_ids)
    except LibraryPaperResolutionError as error:
        raise PaperQAError(str(error)) from error

    searchable: list[str] = []
    searchable_file_ids: list[str] = []
    for resolved_paper in resolved_papers:
        library_paper = resolved_paper.library_paper
        if library_paper is None:
            continue
        azure_file_id = library_paper.get("azure_file_id")
        if (
            library_paper.get("vs_file_status") == "completed"
            and isinstance(azure_file_id, str)
            and azure_file_id
        ):
            searchable.append(resolved_paper.paper_id)
            searchable_file_ids.append(azure_file_id)

    metadata: list[str] = []
    for resolved_paper in resolved_papers:
        paper_id = resolved_paper.paper_id
        paper = resolved_paper.metadata
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
        use_file_search=use_file_search and bool(searchable),
        metadata_paper_ids=paper_ids,
        file_search_paper_ids=searchable,
        file_search_file_ids=searchable_file_ids,
        file_search_file_to_paper_id=dict(zip(searchable_file_ids, searchable)),
    )
    return plan, metadata


def _metadata_synthesis_prompt(*, question: str, metadata_records: list[str]) -> str:
    numbered_records = "\n\n".join(
        f"[{number}]\n{record}"
        for number, record in enumerate(metadata_records, start=1)
    )
    return (
        "Answer using only the selected-paper metadata below. Be explicit about uncertainty.\n"
        "Cite every factual claim drawn from a paper using its evidence number as "
        "[[n]]. Each citation must be its own token: for multiple sources, write "
        "[[1]] [[2]], never [[1],[2]]. Do not cite a number outside the supplied "
        "evidence records.\n\n"
        f"Question:\n{question}\n\n"
        f"Selected-paper metadata:\n{numbered_records}"
    )


def _numbered_metadata_records(metadata_records: list[str]) -> str:
    return "\n\n".join(
        f"[{number}]\n{record}"
        for number, record in enumerate(metadata_records, start=1)
    )


def _file_search_prompt(
    *, question: str, metadata: str, allowed_file_ids: list[str]
) -> str:
    allowlist = "\n".join(f"- {file_id}" for file_id in allowed_file_ids)
    return (
        "Answer the question using only the selected papers listed below.\n\n"
        "IMPORTANT FILE-SCOPE RULES:\n"
        "- The File Search vector store also contains papers the user did not select.\n"
        "- Use a retrieved full-text chunk only when its Azure file_id exactly matches "
        "one of the allowed file IDs below.\n"
        "- Ignore every chunk and citation from any other file, even if it appears relevant.\n"
        "- Before finalizing, verify that every full-text claim and every citation comes "
        "from an allowed file. If the allowed evidence is insufficient, say so rather "
        "than using another file.\n\n"
        "METADATA CITATION RULES:\n"
        "- The selected-paper metadata records below are numbered.\n"
        "- For a claim based on metadata rather than a retrieved PDF chunk, cite it as "
        "[[n]] using that record's number.\n"
        "- Do not use [[n]] for a full-text claim; File Search annotations are the "
        "authoritative citations for those claims.\n"
        "- Use only supplied metadata numbers. For multiple metadata sources, write "
        "separate markers such as [[1]] [[2]].\n\n"
        f"Allowed Azure file IDs:\n{allowlist}\n\n"
        f"Question:\n{question}\n\n"
        f"Metadata for the selected papers:\n{metadata}"
    )


def _append_scope_warning(output: str, unexpected_file_ids: list[str]) -> str:
    """Append a machine-readable marker, matching the papers payload style.

    Frontend should strip this before Markdown render; it is kept in the raw
    message for later inspection.
    """
    payload = json.dumps(
        {
            "unexpected_file_ids": unexpected_file_ids,
            "policy": "soft-citation-check",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{output}\n\n{SCOPE_WARNING_START}{payload}{SCOPE_WARNING_END}"


def answer(
    *,
    user_id: str | None,
    paper_ids: list[str],
    text: str,
    use_file_search: bool = False,
    model: str | None = None,
    trace: SearchV2Trace | None = None,
) -> str:
    plan, metadata_records = build_evidence_plan(
        user_id=user_id,
        paper_ids=paper_ids,
        use_file_search=use_file_search,
    )
    if trace:
        trace.log_synthesis_evidence_plan(
            metadata_paper_ids=plan.metadata_paper_ids,
            file_search_paper_ids=plan.file_search_paper_ids,
            use_file_search=plan.use_file_search,
        )
    settings = get_azure_vector_stores_settings(model=model)
    if not plan.use_file_search:
        input_text = _metadata_synthesis_prompt(
            question=text,
            metadata_records=metadata_records,
        )
        if trace:
            trace.log_synthesis_payload(mode="metadata", input_text=input_text)
        try:
            response = create_text_response(input_text=input_text, settings=settings)
        except AzureVectorStoresError as error:
            raise PaperQAError("Metadata synthesis failed. Please try again.") from error
        output = response_output_text(response)
        if output:
            return replace_numbered_citations(
                output, numbered_paper_ids(plan.metadata_paper_ids)
            )
        raise PaperQAError("Metadata synthesis returned no answer.")
    store = get_user_vector_store(user_id=user_id)
    if store is None or not isinstance(store.get("azure_vector_store_id"), str):
        raise PaperQAError("Selected full texts are not ready. Please retry after uploading them.")
    filters = {"type": "in", "key": "paper_id", "value": plan.file_search_paper_ids}
    input_text = _file_search_prompt(
        question=text,
        metadata=_numbered_metadata_records(metadata_records),
        allowed_file_ids=plan.file_search_file_ids,
    )
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
            settings=settings,
        )
    except AzureVectorStoresError as error:
        raise PaperQAError("Full-text search failed. Please try again.") from error
    output = response_output_text(response)
    if output:
        annotations = response_file_citation_annotations(response)
        resolved_annotations, unexpected_file_ids = resolve_file_annotations(
            annotations, plan.file_search_file_to_paper_id
        )
        cited_file_ids = sorted({annotation["file_id"] for annotation in annotations})
        allowed_file_ids = sorted(plan.file_search_file_to_paper_id)
        if trace:
            trace.log_synthesis_scope_check(
                allowed_file_ids=allowed_file_ids,
                cited_file_ids=cited_file_ids,
                unexpected_file_ids=unexpected_file_ids,
            )
        output = apply_file_citations(output, resolved_annotations)
        output = replace_numbered_citations(
            output, numbered_paper_ids(plan.metadata_paper_ids)
        )
        if unexpected_file_ids:
            return _append_scope_warning(output, unexpected_file_ids)
        return output
    raise PaperQAError("Full-text search returned no answer.")
