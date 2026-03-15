from dotenv import load_dotenv
load_dotenv()
import json
import os
import re
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from service.agent_tools import ALL_AGENT_TOOLS
from service.memory_manager import MemoryManager
from service.query_rewriter import rewrite_query
from service.intent_classifier import classify_intent, Intent
from service.session_state import SESSIONS
import asyncio
from langchain.schema import HumanMessage, AIMessage
from logger_config import get_logger


logging = get_logger()

class NoStopAzureChatOpenAI(AzureChatOpenAI):
    """AzureChatOpenAI that strips/ignores 'stop' for models that don't support it."""
    def _generate(self, messages, stop=None, **kwargs):
        return super()._generate(messages, stop=None, **kwargs)

    def generate(self, messages, stop=None, **kwargs):
        return super().generate(messages, stop=None, **kwargs)

    def generate_prompt(self, prompts, stop=None, **kwargs):
        return super().generate_prompt(prompts, stop=None, **kwargs)

llm = NoStopAzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    streaming=True, 
)

DEFAULT_EMPTY_CONTEXT_MSG = "No documents retrieved."


def get_or_create_chat_session(chat_id):
    """Create or return a per-chat session with a single LLM (no generator)."""

    if chat_id in SESSIONS:
        return SESSIONS[chat_id]

    logging.info("[AgentRunner] Creating new chat session for chat_id=%s", chat_id)

    from langchain.tools import Tool
    import inspect
    import re

    # -----------------------------
    # 1. Create ONE LLM for the session
    # -----------------------------
    llm = get_azure_llm()   # NoStopAzureChatOpenAI

    # -----------------------------
    # 2. Wrap tools per chat
    # -----------------------------
    wrapped_tools = []

    for t in ALL_AGENT_TOOLS:

        # ---- metadata_search ----
        if t.name == "metadata_search":

            def wrapped_metadata_search(filters, user_request=None, _orig=t.func, _cid=chat_id):
                import inspect

                user_text = ""
                try:
                    frame = inspect.currentframe()
                    while frame:
                        if "user_input" in frame.f_locals:
                            user_text = frame.f_locals["user_input"]
                            break
                        frame = frame.f_back
                except Exception:
                    pass

                return _orig(filters=filters, user_request=user_text, chat_id=_cid)

            wrapped_tools.append(
                Tool(
                    name=t.name,
                    func=wrapped_metadata_search,
                    description=t.description
                )
            )

        # ---- semantic_search ----
        elif t.name == "semantic_search":

            def wrapped_semantic_search(query, _orig=t.func, _cid=chat_id):
                import inspect, re
                user_text = query

                try:
                    frame = inspect.currentframe()
                    while frame:
                        if "user_input" in frame.f_locals:
                            user_text = frame.f_locals["user_input"]
                            break
                        frame = frame.f_back
                except Exception:
                    pass

                cleaned_query = re.sub(r"\s{2,}", " ", str(user_text)).strip() or str(query)
                return _orig(query=cleaned_query, chat_id=_cid)

            wrapped_tools.append(
                Tool(
                    name=t.name,
                    func=wrapped_semantic_search,
                    description=t.description
                )
            )

        # ---- select_top_k ----
        elif t.name == "select_top_k":

            def wrapped_select_top_k(count=5, _orig=t.func, _cid=chat_id):
                return _orig(count=count, chat_id=_cid)

            wrapped_tools.append(
                Tool(
                    name=t.name,
                    func=wrapped_select_top_k,
                    description=t.description
                )
            )

        # ---- RAG Q&A tools: inject chat_id ----
        elif t.name == "rag_semantic_qa":
            def _parse_query_question(value):
                """If agent passes one string like 'topic. Question: ...', split into query and question."""
                if isinstance(value, str) and ("Question:" in value or "question:" in value.lower()):
                    parts = re.split(r"\s*[Qq]uestion:\s*", value, maxsplit=1)
                    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (value, value)
                return (value, value) if isinstance(value, str) else (str(value), str(value))

            def wrapped_rag_semantic_qa(query, question=None, _orig=t.func, _cid=chat_id):
                if isinstance(query, dict):
                    payload = query
                    query = payload.get("query", "")
                    if question is None:
                        question = payload.get("question")
                if question is None or (isinstance(question, str) and not question.strip()):
                    query, question = _parse_query_question(query)
                return _orig(query=query, question=question, chat_id=_cid)
            wrapped_tools.append(Tool(name=t.name, func=wrapped_rag_semantic_qa, description=t.description))

        # ---- Other tools unchanged ----
        else:
            wrapped_tools.append(t)

    # -----------------------------
    # 3. Create ReAct tool-calling agent
    # -----------------------------

    core_agent = create_tool_calling_agent(
        llm=llm,
        tools=wrapped_tools,
        prompt=AGENT_PROMPT
    )

    agent = AgentExecutor(
        agent=core_agent,
        tools=wrapped_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        early_stopping_method="generate"
    )


    # -----------------------------
    # 4. Store session
    # -----------------------------
    SESSIONS[chat_id] = {
        "llm": llm,
        "agent": agent,
        "mem": MemoryManager(),
        "_turn_docs": [],
    }

    return SESSIONS[chat_id]
 # 
def is_structured_context(text: str) -> bool:
    if not text or len(text.strip()) < 30:
        return False

    markers = ["**Title:**", "[[ID:", "Authors:", "Year:", "Source:"]
    structural_count = sum(1 for m in markers if m in text)

    if structural_count >= 2:
        return True

    if re.search(r"\[\[ID:\s*[\w\-]+\]\]", text):
        return True
    if re.search(r"^- \*\*Title:\*\* .+", text, flags=re.M):
        return True

    return False


def get_azure_llm():
    return NoStopAzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

def debug(msg: str) -> None:
    """Log a debug message (use logging for production)."""
    logging.debug("%s", msg)


AGENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an intelligent, tool-using research assistant.  
You retrieve academic papers using tools, then write final answers in clear formatting.

Your workflow:
1. Understand the user's request.
2. Decide if retrieval is needed.
3. If needed, call one or more retrieval tools.
4. Once sufficient information (Observations) is available, stop calling tools and generate the **Final Answer**.

You must **never remain silent**. Every user message must produce:
- Either a tool call, or
- A final answer.

====================================================
## 1. TOOL SELECTION POLICY
====================================================

The assistant has two categories of tools:

1. Paper Search Tools → return a LIST of papers
2. RAG Q&A Tools → retrieve papers and generate an ANSWER

The assistant must first determine the user's intent before selecting a tool.

----------------------------------------------------
A. PAPER SEARCH TOOLS
----------------------------------------------------

Use these tools when the user explicitly wants to **discover or list papers**.

Typical user intents:
- finding papers on a topic
- listing papers by author, venue, or year
- browsing search results
- requesting more results from a previous search

Examples of such requests include:
- "find papers on X"
- "show papers about Y"
- "list papers by author Z"
- "give me more papers"

The response should be a **list of papers**, not an explanatory answer.

### semantic_search(query_text)

Use when the user wants papers related to a **topic or concept**, without specifying structured metadata.

Example intents:
- papers about reinforcement learning
- deep learning for NLP
- graph neural networks for recommendation

### metadata_search(filters)

Use when the user provides **structured metadata constraints only** (no topic or keywords):

- title
- authors
- year range (year_min, year_max)
- source / venue
- explicit paper IDs (ids, id_list, paper_ids)

**Do NOT pass topic or keywords to metadata_search.** For topic or keyword queries use semantic_search or mixed_search.

#### metadata_search input rules

These rules must be respected because they match the Python implementation:

1. TITLE
   - Must be a scalar string.
   - If multiple titles are provided, use only the first.

2. AUTHORS
   - A list of authors represents **AND logic**.
   - The paper must contain all listed authors.

3. SOURCE / VENUE
   - Lists are treated as **AND logic**.

4. YEAR RANGE
   - year_min and year_max must be numeric year values.

5. NORMALIZATION
   - Provide literal metadata values.
   - The tool handles normalization internally.

### mixed_search(query_text, filters)

Use when the user provides BOTH:

- a topic description
- structured metadata constraints

Example intent:
"papers about reinforcement learning published in NeurIPS after 2020"

### load_more_papers(chat_id)

Use when the user requests **additional results from the current search**.

Typical intents:
- "show more"
- "load more papers"
- "next page"

When pagination is requested, **only call load_more_papers**.
Do not run a new semantic_search or metadata_search.

----------------------------------------------------
B. RAG QUESTION-ANSWERING TOOLS
----------------------------------------------------

Use these tools when the user asks a **question that should be answered using papers**.

The goal is to:
1. retrieve relevant papers
2. synthesize an answer based on them

The response should be an **explanation or analysis**, not a list of papers.

### rag_semantic_qa(query, question, chat_id)

Use when the question is **topic-based** and contains no structured metadata filters.

Example intents:
- "What methods are used in RAG research?"
- "What datasets are used in attention models?"

For questions about papers identified by **metadata** (author, year, venue, title): use **metadata_search(filters)** first to retrieve that set of papers; then answer the user's question from the tool output.

Example intents:
- "What datasets were used in CHI 2023 papers?" → metadata_search with year_min=2023, venues=[\"CHI\"]
- "What is X in this paper [title]?" → metadata_search with title filter, then answer from results.

For questions that combine **topic + metadata** (e.g. "What do CHI papers say about usability?"): use **mixed_search(query_text, filters)** first to retrieve papers; then answer the user's question from the tool output.

----------------------------------------------------
C. WHEN NOT TO USE RETRIEVAL
----------------------------------------------------

Do NOT use retrieval tools when the question is:

1. purely conceptual  
   Example: "What is RAG?"

2. answerable from general knowledge

3. a clarification request

4. rewriting or editing previous text

5. a follow-up that relies only on conversation context

====================================================
## 2. RULES FOR TOOL CALLING
====================================================
- Only **ONE tool call per assistant message**.
- Tool call must be valid JSON.
- **INTENT_HINT:** When the user message includes a line like `[INTENT_HINT: intent=..., tool=..., slots=...]`, treat it as a routing hint from the system. Prefer calling the suggested **tool** (e.g. metadata_search, semantic_search, load_more_papers) for your first tool call unless it clearly does not fit the request. Use **slots** to shape your tool arguments when relevant (e.g. authors, year_min, topic).
- Never hallucinate metadata fields.
- Never combine multiple titles.
- Never guess missing authors or titles.
- After the tool gives results, write the **Final Answer**.

====================================================
## 3. HANDLING CONSTRAINTS (CRITICAL)
====================================================
When the user specifies constraints (e.g., "after 2010", “by Goodfellow”):

### ✔ If retrieved papers satisfy the constraints  
→ Filter them in your Final Answer.  
→ NEVER hallucinate missing metadata.

### ✔ If retrieved papers DO NOT satisfy constraints  
(e.g., the tool returns mostly pre-2010 papers)

You MUST:
1. Still produce a Final Answer.
2. Say clearly:
   - “No papers fully satisfy your constraint.”
3. Then show the closest matching retrieved papers (do not hide them).
4. DO NOT call more tools unless the user explicitly asks.

### ✔ If the tool returns:  
{{"SYSTEM_NOTICE": "NO_MATCH", ...}}
You MUST:
- Stop calling tools  
- Write a Final Answer:
  - Apologize briefly
  - Explain no matches found
  - Suggest how to broaden the query

This rule avoids tool-calling loops and agent paralysis.

====================================================
## 4. HOW TO PARSE USER REQUESTS
====================================================

### ✔ Authors list detection
The following MUST produce an authors list:
- “A and B”
- “A & B”
- “A, B, and C”
- “A, B”
- “papers by A together with B”

→ Convert these into:
{{"authors": ["A", "B"]}}

### ✔ Keywords list detection
Same logic as authors.

### ✔ Paper ID detection (CRITICAL)
If the user refers to specific papers with stable IDs or a numbered list, such as:
- "Explain this paper (ID: 64268)"
- "Explain both of these papers: 1. <Title A> (ID: 64268), 2. <Title B> (ID: 55391)"

You MUST:
- Prefer **metadata_search** when the question is only about those specific papers.
- Use **mixed_search** when the user both specifies IDs and also describes a broader topic or open-ended requirement.
- Construct filters that include these IDs, e.g. a filters object with key "ids" mapped to ["64268", "55391"] or key "id_list" mapped to ["64268", "55391"].
- **Do NOT** call pure semantic_search for ID-based follow-ups.

### ✔ Title parsing
If the user provides multiple titles:
- Use only the **first** one
- Ignore the rest

### ✔ Key:value fallback parsing
If the user types:
"authors: A AND B"

→ Your job:
- Detect this pattern
- Convert into filters={{"authors": ["A", "B"]}}


====================================================
## 5. FINAL ANSWER FORMAT (MANDATORY)
====================================================

When no tool is being called, the assistant MUST follow the format rules below.

----------------------------------------------------
A. Returning Papers (List Results)
----------------------------------------------------

If the user asked to search for papers and the result comes from:
- semantic_search
- metadata_search
- load_more_papers

Then the assistant MUST:

1. Start with one or two natural sentences introducing the list.
   Example:
   "Here are some papers related to reinforcement learning:"
   "Here are additional papers from this search:"

2. Output ALL papers returned by the tool.
   - Do NOT omit any paper.
   - Do NOT reorder papers.

3. Use the **compact display format** below.

Compact display format:

- **Title:** <title> [[ID:<paper id>]]
  Authors: <authors>
  Year: <year>
  Source: <source>

4. Do NOT include Abstract or Score unless the user explicitly asks for:
   - abstracts
   - relevance scores
   - explanations of relevance

5. If the tool output contains the tag:

[SIGNAL:SHOW_LOAD_MORE]

Then copy the tag exactly and place it on a new line at the very end of the answer.

Do NOT modify, explain, or paraphrase this tag.

----------------------------------------------------
B. Summary / Comparison Questions
----------------------------------------------------

If the user asks for explanations, summaries, or comparisons:

- Provide a clear and concise explanation.
- Cite relevant papers using the format:

*Paper Title* [[ID:<paper id>]]

Only cite papers that are relevant to the user's question.

----------------------------------------------------
C. Relevance Rules (VERY IMPORTANT)
----------------------------------------------------

The assistant MUST only reference papers that are clearly relevant to the user's question.

If retrieved papers are:
- weakly related
- unrelated
- insufficient to support the answer

Then the assistant MUST answer the question **without citing those papers**.

----------------------------------------------------
D. No Relevant Papers
----------------------------------------------------

If the user requested a paper search but:

- no papers were retrieved, OR
- none of the retrieved papers meet the user's requirements

Then respond with:

"No papers were found that meet your requirements."

Do NOT fabricate papers.

----------------------------------------------------
E. Example (Correct Output)
----------------------------------------------------

Here are the relevant papers:

- **Title:** Referential Choices in Discourse [[ID:12345]]
  Authors: Alyssa Ibarra, J. Smith
  Year: 2012
  Source: CogSci

- **Title:** Another Paper Title [[ID:67890]]
  Authors: Jane Doe
  Year: 2020
  Source: ACL

[SIGNAL:SHOW_LOAD_MORE]

====================================================
## 6. SAFETY RULES
====================================================
- Never invent IDs, authors, years, sources, or titles.
- Only reference papers explicitly returned by tools or memory.
- Always produce either a tool call or a final answer.

====================================================
""",
    ),
    # ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])



def reset_session(chat_id: str):
    """Clear conversation and docs for a single chat tab."""
    if chat_id not in SESSIONS:
        return

    sess = SESSIONS[chat_id]

    # Reset sliding-window memory
    if "mem" in sess:
        sess["mem"].clear()
    else:
        sess["mem"] = MemoryManager()   # fallback

    # Reset per-turn docs
    sess["_turn_docs"] = []

    logging.info("[Session] Cleared chat session %s", chat_id)


def reset_all_sessions() -> None:
    """Clear all chat sessions completely."""
    SESSIONS.clear()
    logging.info("[Session] All sessions cleared.")


def log_rag(tag: str, msg: str) -> None:
    """Log a RAG-related message."""
    logging.info("[%s] %s", tag, msg)


def _resolve_selected_papers_from_cache(chat_id: str, requested: list, question: str):
    """
    Look up requested papers (list of (title, id)) in session cache only.
    Returns (resolved_agent_input_str, from_cache: bool).
    """
    from service.rag_core import get_session_docs, format_docs

    requested_ids = {str(pid).strip() for (_, pid) in requested if pid}
    requested_titles = {str(t).strip().lower() for (t, _) in requested if t}

    docs = get_session_docs(chat_id)
    matched = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        if isinstance(d, dict):
            md = d.get("metadata", d)
        did = str(md.get("id") or md.get("ID") or "").strip()
        title = str(md.get("title") or "").strip().lower()
        if did and did in requested_ids:
            matched.append(d)
        elif title and title in requested_titles:
            matched.append(d)

    if matched:
        formatted = format_docs(matched, include_abstract=True, include_score=False)
        resolved = (
            "The user has selected the following papers from the current session. "
            "Answer the question using ONLY these papers. Do NOT call metadata_search, semantic_search, or mixed_search.\n\n"
            "Papers:\n"
            f"{formatted}\n\n"
            f"Question: {question}"
        )
        logging.info(f"[agent] Selected papers resolved from cache ({len(matched)} docs). No retrieval.")
        return resolved, True

    ids = [i for (_, i) in requested if i]
    ids_json = json.dumps(ids)
    resolved = (
        "The user asked about specific papers that are NOT in the session cache. "
        "You MUST call metadata_search with filters containing the paper IDs (use \"ids\" or \"id_list\"), then answer. "
        "Do NOT call semantic_search or mixed_search.\n\n"
        "Call hint: metadata_search(filters with an \"ids\" field set to the requested IDs), chat_id=...\n\n"
        f"Requested paper IDs: {ids_json}\n\n"
        f"Question: {question}"
    )
    logging.info("[agent] Selected papers NOT in cache; agent must use metadata_search with ids.")
    return resolved, False


from langchain.schema import HumanMessage, AIMessage
def convert_history_text_to_messages(history_text: str):
    messages = []
    lines = history_text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("USER:"):
            messages.append(HumanMessage(content=line[len("USER:"):].strip()))
        elif line.startswith("ASSISTANT:"):
            messages.append(AIMessage(content=line[len("ASSISTANT:"):].strip()))
        else:
            # fallback — treat as human
            messages.append(HumanMessage(content=line))
    return messages


async def _fallback_direct_answer(
    llm, user_input: str, mem: MemoryManager = None, session: dict = None
):
    """
    When any step or tool fails, stream a direct LLM answer to the user's question.
    Yields content chunks; optionally updates memory if mem provided.
    """
    try:
        messages = [HumanMessage(content=user_input)]
        full = []
        if hasattr(llm, "astream") and callable(getattr(llm, "astream")):
            async for chunk in llm.astream(messages):
                if getattr(chunk, "content", None):
                    full.append(chunk.content)
                    yield chunk.content
        else:
            resp = llm.invoke(messages)
            text = getattr(resp, "content", str(resp))
            full.append(text)
            yield text
        if mem and full:
            mem.add_turn("assistant", "".join(full))
    except Exception as e:
        logging.warning("[fallback_direct_answer] LLM fallback failed: %s", e)
        yield "I'm sorry, I couldn't complete your request. Please try again or rephrase your question."


async def run_two_stage_rag_stream(
    user_input: str,
    chat_id: str = "default",
    selected_paper_ids: list = None,
    selected_paper_titles: list = None,
):
    session = None
    try:
        session = get_or_create_chat_session(chat_id)
    except Exception as e:
        logging.warning("[run_two_stage_rag_stream] session creation failed: %s", e)
        llm = get_azure_llm()
        async for chunk in _fallback_direct_answer(llm, user_input, mem=None, session=None):
            yield chunk
        return

    agent = session["agent"]
    mem: MemoryManager = session["mem"]
    llm = session["llm"]
    session["_turn_docs"] = []
    mem.add_turn("user", user_input)

    # ── Selected papers: resolve from cache or force metadata_search (no rewrite) ──
    agent_input_text = user_input
    if selected_paper_ids or selected_paper_titles:
        try:
            ids = list(selected_paper_ids) if selected_paper_ids else []
            titles = list(selected_paper_titles) if selected_paper_titles else []
            n = max(len(ids), len(titles)) or 0
            requested = [
                (titles[i] if i < len(titles) else None, ids[i] if i < len(ids) else None)
                for i in range(n)
            ]
            if requested:
                agent_input_text, _ = _resolve_selected_papers_from_cache(
                    chat_id, requested, user_input.strip()
                )
        except Exception as e:
            logging.warning("[run_two_stage_rag_stream] resolve selected papers failed: %s", e)
            agent_input_text = user_input
    else:
        # ── L1: Normalise ──
        clean_input = user_input.strip()[:2000]

        # ── L2: Rewrite against history ──
        try:
            rewrite = rewrite_query(clean_input, llm, session=session, chat_id=chat_id)
            clean_input = rewrite.rewritten_query
            logging.info(
                f"[L2] rewrite={rewrite.was_rewritten} "
                f"type={rewrite.rewrite_type} query='{clean_input[:80]}...'"
            )
        except Exception as e:
            logging.warning(f"[L2] rewrite failed: {e}")
            clean_input = user_input.strip()[:2000]

        # ── L3: Classify intent ──
        try:
            intent = classify_intent(clean_input, llm, session=session, chat_id=chat_id)
            logging.info(
                f"[L3] intent={intent.intent} conf={intent.confidence:.2f} "
                f"hint={intent.tool_hint}"
            )
        except Exception as e:
            logging.warning(f"[L3] classify failed: {e}")
            intent = None

        # ── L4: Confidence router (short-circuit) ──
        if intent and intent.needs_clarification and intent.clarification_question:
            yield intent.clarification_question
            mem.add_turn("assistant", intent.clarification_question)
            return

        if intent and intent.intent == Intent.SMALL_TALK:
            try:
                response = llm.invoke([HumanMessage(content=clean_input)]).content
                yield response
                mem.add_turn("assistant", response)
                return
            except Exception as e:
                logging.warning("[run_two_stage_rag_stream] SMALL_TALK LLM failed: %s", e)
                async for chunk in _fallback_direct_answer(llm, clean_input, mem=mem, session=session):
                    yield chunk
                return

        # ── L5: Build agent input with intent hint ──
        try:
            if intent:
                intent_hint = (
                    f"\n[INTENT_HINT: intent={intent.intent.value}, "
                    f"tool={intent.tool_hint}, "
                    f"slots={intent.slots}]"
                )
                agent_input_text = clean_input + intent_hint
            else:
                agent_input_text = clean_input
        except Exception as e:
            logging.warning("[run_two_stage_rag_stream] L5 intent hint failed: %s", e)
            agent_input_text = clean_input

    # ── Agent execution + streaming ──
    final_answer = ""
    agent_input = {"input": agent_input_text}

    try:
        async for event in agent.astream_events(agent_input, version="v1"):
            kind = event["event"]

            if kind == "on_tool_end":
                out = event["data"].get("output")
                tool_name = event.get("name", "")

                # Only for load_more_papers: return tool output directly (no LLM call). All other tools go through the model.
                if tool_name == "load_more_papers" and isinstance(out, str) and out.strip():
                    final_answer += out
                    logging.info("[agent] load_more_papers: returning tool output directly (no LLM call).")
                    yield out

                # mixed_search dict output
                if isinstance(out, dict):
                    docs = out.get("final_docs") or out.get("semantic_docs") or out.get("metadata_docs") or []
                    session["_turn_docs"].extend(docs)

                # semantic_search / metadata_search list output
                elif isinstance(out, list):
                    session["_turn_docs"].extend(out)

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content and not chunk.tool_call_chunks:
                    final_answer += chunk.content
                    yield chunk.content

        asyncio.create_task(async_update_memory(session, final_answer))
    except Exception as e:
        logging.warning("[run_two_stage_rag_stream] agent/tool failed: %s", e)
        async for chunk in _fallback_direct_answer(llm, user_input, mem=mem, session=session):
            yield chunk

async def async_update_memory(session, final_text: str):
    """Update memory after turn completes (sliding window only)."""
    try:
        mem: MemoryManager = session["mem"]

        # store assistant reply
        mem.add_turn("assistant", final_text)

        # store retrieved docs
        if session["_turn_docs"]:               
            mem.set_docs(session["_turn_docs"])    
        
        else:
            debug("No docs retrieved this turn.")

        # clear turn doc buffer
        session["_turn_docs"] = []

        debug("=== Memory Update End ===")

    except Exception as e:
        log_rag("Memory", f"Async memory update FAILED: {e}")


__all__ = [
    "run_two_stage_rag_stream",
    "reset_session",
    "reset_all_sessions",
]