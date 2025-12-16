from dotenv import load_dotenv
load_dotenv()
import os
import re
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor, initialize_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from service.agent_tools import ALL_AGENT_TOOLS
from prompt import RESPONSE_TEMPLATE
from service.memory_manager import MemoryManager
import asyncio

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

# 🔐 Per-tab session state (conversation + docs + agent + generator)
SESSIONS = {}  # { chat_id: {"memory": ..., "docs": [...], "agent": ..., "generator": ...} }

def get_or_create_chat_session(chat_id):
    """Create or return a per-chat session with a single LLM (no generator)."""

    if chat_id in SESSIONS:
        return SESSIONS[chat_id]

    print(f"[AgentRunner] 🧠 Creating new chat session for chat_id={chat_id}")

    from langchain.tools import Tool
    import inspect
    import re

    llm = get_azure_llm()   # NoStopAzureChatOpenAI
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
                except:
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
                except:
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

        # ---- Other tools unchanged ----
        else:
            wrapped_tools.append(t)

    # -----------------------------
    # 4. Create ReAct tool-calling agent
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

    SESSIONS[chat_id] = {
        "llm": llm,
        "agent": agent,
        "mem": MemoryManager(),
        "_turn_docs": [],
    }

    return SESSIONS[chat_id]


def is_structured_context(text: str) -> bool:
    """
    判断一段 Agent 输出是否为结构化检索结果（即可安全作为上下文使用）
    """
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

def debug(msg: str):
    print(f"[DEBUG] {msg}")


from langchain.prompts import ChatPromptTemplate
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
## 1. WHEN TO USE EACH TOOL
====================================================
### semantic_search(query_text)
Use when the user describes a *topic*, *concept*, or *natural-language description*  
(e.g., "papers on data mining", "deep learning for NLP").

### metadata_search(filters)
Use when the user specifies *structured filters*:
- title
- author
- year_min / year_max
- source

### 💡 VERY IMPORTANT — metadata_search input rules
These rules MUST be followed because they match the Python implementation:

1. **TITLE**
   - Title must be **scalar only**
   - If the user provides a list of titles, USE ONLY THE FIRST ONE.
   - Never include multiple titles in one metadata_search call.

2. **AUTHORS**
   - A list of authors means **AND logic**
   - The paper must contain **all** listed authors
   - Examples:
        "A and B" → ["A", "B"]
        "A, B, and C" → ["A", "B", "C"]

3. **SOURCE / VENUE**
   - List = **AND logic**
   - In practice usually only one source is used
   - But if list is provided → treat as AND logic

4. **YEAR RANGE**
   - year_min / year_max must be numeric year values

5. **Normalization**
   - Your job is to provide clean, literal metadata values
   - The tool will normalize punctuation and casing itself

### semantic_search(query_text)
Use when the user describes:
- keywords
- topics
- research areas
- open-ended descriptions

### mixed_search(query_text, filters)
Use when BOTH:
- the user provides metadata filters AND
- a topic or natural-language description 

### DO NOT use retrieval if:
- The question is conceptual (“What is RAG?”, “Explain transformers”)
- The answer is clearly available in conversation memory
- The user is asking for clarification, summarization, rewriting, or follow-up questions

====================================================
## 2. RULES FOR TOOL CALLING
====================================================
- Only **ONE tool call per assistant message**.
- Tool call must be valid JSON.
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

When not calling tools, follow this format exactly.

### ✔ When returning a LIST of papers:
- **Title:** <title> [[ID:<paper id>]]
  Authors: <authors>
  Year: <year>
  Source: <source>

### ✔ When providing a SUMMARY or COMPARISON:
- Explain clearly and concisely.
- Cite papers using: *Title* [[ID:<id>]].

### ✔ Example:
Here are the relevant papers:
- **Title:** Referential Choices in Discourse [[ID:12345]]
  Authors: Alyssa Ibarra, J. Smith  
  Year: 2012  
  Source: CogSci

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

CONTEXTUAL_REWRITE_PROMPT = PromptTemplate.from_template("""

You are a cautious and precise assistant that rewrites user questions
ONLY when there is *clear and unambiguous* contextual reference in the conversation.

Your goal is to:
- Make the final rewritten question self-contained.
- Resolve vague references **ONLY when you are 100% sure** what the user refers to.
- NEVER guess, infer, or hallucinate missing details.
- NEVER force rewriting if context is incomplete or ambiguous.

Conversation so far:
{chat_history}

Latest user question:
{current_query}

============================================================
### STRICT RULES FOR REWRITING (CRITICAL)
============================================================

1. ✔ Rewrite ONLY IF:
   - The user clearly refers to papers previously shown (e.g., "the first one", "the survey", "that KDD paper").
   - You can unambiguously map these references to EXACT titles/authors/IDs.

2. ✔ If the reference is ambiguous (for example):
   - Multiple papers could match
   - The assistant's earlier message did not contain a specific paper list
   - Titles are truncated or partially shown
   - The conversation mentions a topic but no clear paper list

   → DO NOT rewrite.

3. ✔ If you cannot confidently rewrite:
   → Return exactly the line below (no changes):
     UNSURE_REWRITE

   (This triggers a safe path in the agent, so the agent can ask the user for clarification.)

4. NEVER invent:
   - Paper titles
   - Authors
   - Years
   - IDs
   - Assumptions about “the first paper” without certainty

============================================================
### Output Format
============================================================

- If confident → output ONLY the rewritten self-contained question.
- If NOT confident → output ONLY:
  UNSURE_REWRITE

Do NOT include explanations.

Now, produce the correct output.
""")


DECIDE_PROMPT = """
Does the user's new query refer to *previously mentioned papers or content*?
Answer yes or no.

Conversation:
{chat_history}

User query:
{current_query}

Answer only: yes or no
"""


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

    print(f"[Session] Cleared chat session {chat_id}")


def reset_all_sessions():
    """Clear all chat sessions completely."""
    SESSIONS.clear()
    print("[Session] All sessions cleared.")


from langchain.schema import HumanMessage, AIMessage
def rewrite_query_with_context(llm, history_text: str, current_query: str) -> str:
    if not history_text.strip():
        return current_query

    chain = LLMChain(llm=llm, prompt=CONTEXTUAL_REWRITE_PROMPT)
    resp = chain.invoke({
        "chat_history": history_text,
        "current_query": current_query
    })

    rewritten = (
        resp.get("text")
        if isinstance(resp, dict)
        else str(resp)
    ).strip()

    # If model is unsure → use original query
    if rewritten.upper().startswith("UNSURE"):
        return current_query

    return rewritten or current_query


def should_rewrite(llm, history_text: str, current_query: str) -> bool:
    """Return True if the query refers to previous conversation."""
    if not history_text.strip():
        return False

    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(DECIDE_PROMPT))
    res = chain.invoke({
        "chat_history": history_text,
        "current_query": current_query
    })

    answer = (res.get("text") if isinstance(res, dict) else str(res)).strip().lower()
    return answer.startswith("y")


def log_rag(tag: str, msg: str):
    print(f"[{tag}] {msg}")


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


async def run_two_stage_rag_stream(user_input: str, chat_id: str = "default"):
    session = get_or_create_chat_session(chat_id)
    agent = session["agent"]
    mem: MemoryManager = session["mem"]
    llm = session["llm"]

    # reset per-turn doc buffer
    session["_turn_docs"] = []

    # add user turn (sliding window)
    mem.add_turn("user", user_input)

    # ----------- Rewriting phase -----------
    history_text = mem.get_history_text()
    debug(f"History before rewrite:\n{history_text}")  ## debug

    try:
        use_rewrite = should_rewrite(llm, history_text, user_input)
        debug(f"Should rewrite? {use_rewrite}")

        rewritten_query = (
            rewrite_query_with_context(llm, history_text, user_input)
            if use_rewrite else user_input
        )
        debug(f"Rewritten query: {rewritten_query}")    ## debug

    except:
        # debug(f"Rewrite failed: {e}")                 ## debug
        rewritten_query = user_input

    # ----------- Agent Execution + Streaming -----------
    final_answer = ""

    history_text = mem.get_history_text()
    history_messages = convert_history_text_to_messages(history_text)

    agent_input = {
        "input": rewritten_query,
        # "chat_history": history_messages
    }

    async for event in agent.astream_events(agent_input, version="v1"):
        kind = event["event"]

        if kind == "on_tool_end":
            out = event["data"].get("output")

            # mixed_search dict output
            if isinstance(out, dict):
                docs = out.get("final_docs") or out.get("semantic_docs") or out.get("metadata_docs") or []

                debug(f"Tool returned dict with {len(docs)} docs")   # debug
                for d in docs:                                       # debug
                    debug(f"  Added doc ID={d.get('ID')} Title={d.get('Title')}")  # debug

                session["_turn_docs"].extend(docs)

            # semantic_search / metadata_search list output
            elif isinstance(out, list):

                debug(f"Tool returned list with {len(out)} docs")        # debug
                for d in out:                                            # debug
                    debug(f"  Added doc ID={d.get('ID')} Title={d.get('Title')}")   # debug
                
                session["_turn_docs"].extend(out)

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content and not chunk.tool_call_chunks:
                final_answer += chunk.content
                yield chunk.content

    asyncio.create_task(async_update_memory(session, final_answer))
    # return final_answer



async def async_update_memory(session, final_text: str):
    """Update memory after turn completes (sliding window only)."""
    try:
        mem: MemoryManager = session["mem"]

        debug("=== Memory Update Start ===")               # debug
        debug(f"Assistant final turn text:\n{final_text}") # debug

        # store assistant reply
        mem.add_turn("assistant", final_text)

        # store retrieved docs
        if session["_turn_docs"]:

            debug(f"Saving {len(session['_turn_docs'])} docs into memory")    # debug
            for d in session["_turn_docs"]:                                   # debug
                if isinstance(d, dict):                                       # debug
                    debug(f"  Doc ID={d.get('ID')} Title={d.get('Title')}")   # debug
                elif hasattr(d, "metadata"):                                  # debug
                    debug(f"  Doc ID={d.metadata.get('ID')} Title={d.metadata.get('Title')}")  # debug
                else:                                                         # debug
                    debug(f"  Doc (unknown format): {d}")                     # debug
                
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