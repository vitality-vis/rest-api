---
name: summarize-selected-papers
description: Summarizes the user's current paper selection adaptively. Use when the user asks to summarize, synthesize, compare, or give an overview of selected papers without specifying a rigid output format.
---

# Summarize selected papers

Lead with the useful summary. Do not narrate the workflow or tools.

## Length and shape (default)

Keep the answer **short by default**—about half a screen unless the user asks for more.

- **Thematic overview** (default for roughly 9+ papers, and fine for smaller sets): a few short paragraphs or bullets on shared themes, contrasts, and useful takeaways. Cite a handful of representative papers as `[[ID:…]]`. **Do not** write one section per paper.
- **Paper-by-paper** only when the user explicitly asks for it (e.g. 逐篇 / each paper / one by one).
- Roughly 2–8 papers: still prefer a compact synthesis; mention individuals only where they matter.
- More than roughly 30 papers: landscape of major themes only; do not walk the list.

Do not pad with methods/findings/limitations templates or a generic closing offer.

## Evidence

- If this turn already includes **Preloaded selected-paper evidence**, use it and **do not** call `inspect_selected_papers` or `get_selected_paper_evidence` unless the preload is insufficient or you need a different subset/group.
- Otherwise: for a large or unclear selection, `inspect_selected_papers` first; then `get_selected_paper_evidence` once with the needed detail.
- Default to titles/abstracts. Use `search_library_full_text` or `get_selected_paper_full_text` only when the request needs details abstracts unlikely to provide and full text is available.
- Do not infer content from IDs. Disclose sampling, truncation, or missing full text briefly.
- Cite factual claims as `[[ID:paper_id]]` using only IDs from preload or tools. Never `[[doi:…]]` or other bracket forms.
