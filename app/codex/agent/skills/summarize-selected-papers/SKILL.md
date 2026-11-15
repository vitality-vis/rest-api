---
name: summarize-selected-papers
description: Summarizes the user's current paper selection adaptively. Use when the user asks to summarize, synthesize, compare, or give an overview of selected papers without specifying a rigid output format.
---

# Summarize selected papers

Use the selected-paper evidence tools and generate the final answer yourself. Lead with the substantive summary, not a description of the workflow or tools you are about to use. Do not announce that you are applying a summarization workflow, mapping the scope, or retrieving evidence.

## Evidence

- If a large selection's scope is unclear, call `inspect_selected_papers` first.
- Call `get_selected_paper_evidence` with the detail and subset needed for the request.
- Do not infer paper content from IDs.
- If evidence is sampled, truncated, or title-only, disclose the limitation briefly.
- Cite factual paper claims as `[[ID:paper_id]]` using only IDs returned by the tool. Never use `[[doi:…]]` or other bracket citation forms.

## Adapt the summary

- One paper: explain its core question, approach, findings, and useful details.
- Roughly 2–8 papers: synthesize and compare meaningful connections, differences, and newer developments when supported by evidence.
- Roughly 9–30 papers: foreground major themes and patterns; use only a few representative papers as examples.
- More than roughly 30 papers: provide a high-level research landscape centered on major themes; do not summarize papers one by one.

Choose the clearest format for the evidence and request. Do not force methods, findings, and differences into separate sections. Do not make the response longer merely because more papers were selected. Keep the answer focused by default, while allowing useful detail for small selections or explicit user requests.
