# Markdown formatting guidelines (used across all prompts)
MARKDOWN_FORMAT_GUIDE = """
## Markdown Formatting Guidelines (IMPORTANT)
Your need to produce content in **strict, clean, well-formatted, raw Markdown format**. Never mix formatting syntax.

Formatting rules (must follow):

- Use Markdown syntax only.
- Headings:
   - Use # for h1, ## for h2, ### for h3
   - NEVER add ** around headings
   - NEVER add — after headings
- Bold text: Use ** only for emphasis within paragraphs, NOT for headings.
- List items: Start with "- " (dash + space)
- Bold lead-in phrases:
   - For key points, you may start with a bold phrase: "**Brief descriptive phrase.** [detailed text...]"
   - Keep it short (3-8 words), not a full sentence
   - Examples: "**They support early-stage ideation.**", "**Mixed or ambiguous effects.**"
   - Use selectively to highlight major findings or contrasts
   - Maintain natural variation—not every paragraph needs this
- Complete each section fully before moving to the next.
- Preserve proper blank lines between sections.
- Do NOT render formatting.
- Do NOT explain the formatting.
- Do NOT include any text outside the Markdown content.

Output must be directly copy-pasteable into a Markdown (.md) file.
"""

# Citation instructions (used in summarize and literature review prompts)
CITATION_INSTRUCTIONS = """
## Citation Rules (IMPORTANT)
1. When using information from papers, you MUST cite the specific sentence using format [X.Y]
   - X: paper number (starting from 0)
   - Y: sentence number (starting from 0)
2. Multiple citations
   - Format: [0.1][0.2] or [0.1][1.3]
   - NEVER use ranges like [0.1-0.3] or [3.1-3.2]
   - NEVER use commas inside brackets like [0.1, 0.2]
   - Each citation must be in its own separate brackets
3. Citation examples:
   - "Visual analytics combines automated analysis with interactive visualization [0.1]"
   - "The approach enables better decision-making [0.2]"
   - "Another study found that [1.0]"
4. If citing the overall paper (not a specific sentence), use [X] (paper number only)
5. When referring to specific papers in the main text, NEVER use "Paper 0", "Paper 1", "Paper X" in your writing
   - Wrong: "Paper 0 shows that..."
   - Correct: "Johnson et al. show that... [0.3]", "according to Johnson et al., etc."

Write your response directly with inline citations (not in JSON format).
"""

SUMMARIZE_PROMPT = f"""
{{prompt}}

{CITATION_INSTRUCTIONS}

{MARKDOWN_FORMAT_GUIDE}

content: \n
{{content}}
"""

LITERATURE_REVIEW_PROMPT = f"""
{{prompt}}

{CITATION_INSTRUCTIONS}

{MARKDOWN_FORMAT_GUIDE}

content: \n
{{content}}
"""
