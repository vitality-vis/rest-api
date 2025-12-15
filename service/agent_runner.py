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

    # -----------------------------
    # 1. Create ONE LLM for the session
    # -----------------------------
    llm = get_azure_llm()   # NoStopAzureChatOpenAI

    # -----------------------------
    # 2. Create memory buffer
    # -----------------------------
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True
    # )

    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True,
    #     output_key="output"  # <--- ADD THIS LINE
    # )

    # -----------------------------
    # 3. Wrap tools per chat
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

    # agent = AgentExecutor(
    #     agent=core_agent,
    #     tools=wrapped_tools,
    #     memory=memory,
    #     verbose=True,
    #     handle_parsing_errors=True,
    #     return_intermediate_steps=True,
    #     early_stopping_method="generate"
    # )

    agent = AgentExecutor(
        agent=core_agent,
        tools=wrapped_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        early_stopping_method="generate"
    )

    # -----------------------------
    # 5. Store session (NO GENERATOR ANYMORE)
    # -----------------------------
    SESSIONS[chat_id] = {
        "llm": llm,
        "agent": agent,
        "mem": MemoryManager(),
        "_turn_docs": [],
        "docs": [],  # Required for metadata_search to work
    }

    return SESSIONS[chat_id]

# =====================================================
# 🧩 通用上下文验证器
# =====================================================
def is_structured_context(text: str) -> bool:
    """
    判断一段 Agent 输出是否为结构化检索结果（即可安全作为上下文使用）
    """
    if not text or len(text.strip()) < 30:
        return False

    # 关键结构标记
    markers = ["**Title:**", "[[ID:", "Authors:", "Year:", "Source:"]
    structural_count = sum(1 for m in markers if m in text)

    # 规则1：至少两个结构标记
    if structural_count >= 2:
        return True

    # 规则2：存在常见的结果列表模式
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

### ✔ semantic_search(query_text)
Use when the user describes a *topic*, *concept*, or *natural-language description*  
(e.g., "papers on data mining", "deep learning for NLP").

### ✔ metadata_search(filters)
Use when the user specifies *structured filters*:
- title
- author
- year_min / year_max
- source
- keywords

### ✔ mixed_search(query_text, filters)
Use when the request includes **both topic AND metadata constraints**  
(e.g., "data mining papers after 2010", "LLM papers by Bengio").

→ The tool internally merges semantic + metadata results.  
→ If no overlapping results, the tool will include a `SYSTEM_NOTICE` field.

### ✔ select_top_k(k)
Use only after a retrieval tool, to limit the list.

### ✔ recall_memory(reference)
Use only when the user refers to previously returned papers  
(e.g., “the first one”, “the survey you just mentioned”).

### ❌ DO NOT use retrieval if:
- The question is conceptual (“What is RAG?”, “Explain transformers”)
- The answer is clearly available in conversation memory
- The user is asking for clarification, summarization, rewriting, or follow-up questions

====================================================
## 2. RULES FOR TOOL CALLING
====================================================

- Tool calls MUST be valid JSON (no code blocks).
- One tool call per message.
- Never hallucinate metadata: only use what tools return.
- When multiple papers are requested, retrieve them in a single call (no loops).
- After retrieving results, **stop using tools** and produce the Final Answer.

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
## 4. FINAL ANSWER FORMAT (MANDATORY)
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
## 5. SAFETY RULES
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
You are a helpful assistant that rewrites user questions to make them self-contained and context-aware.

Conversation so far:
{chat_history}

Latest user question:
{current_query}

Rewrite the question so that it explicitly includes the correct references from the conversation.
- Replace vague phrases like "the first paper", "that one", "the survey", etc. with the actual paper titles, authors, or topics mentioned before.
- Keep the question concise and natural.
- If there is no relevant context, return the query unchanged.

### Examples
Conversation:
User: show me papers about neural machine translation
Assistant: 1. Attention is All You Need (Vaswani et al., 2017)
User: explain the first paper
→ Rewritten: explain the paper "Attention is All You Need" by Vaswani et al. (2017)

Conversation:
User: list some works about data mining
Assistant: 1. Data Mining Techniques (Han, 1996)
User: summarize that paper
→ Rewritten: summarize the paper "Data Mining Techniques" by Jiawei Han (1996)

Conversation:
User: what is retrieval-augmented generation
Assistant: explanation text
User: what about hybrid models
→ Rewritten: what are hybrid retrieval-augmented generation models

Now rewrite the latest question using the same style.
Output only the rewritten question.
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

    print(f"[Session] 🧹 Cleared chat session {chat_id}")


def reset_all_sessions():
    """Clear all chat sessions completely."""
    SESSIONS.clear()
    print("[Session] 🔄 All sessions cleared.")


from langchain.schema import HumanMessage, AIMessage
def rewrite_query_with_context(llm, history_text: str, current_query: str) -> str:
    """Rewrite user query using recent sliding-window context."""
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

                debug(f"Tool returned dict with {len(docs)} docs")   ## debug
                for d in docs:                                       ## debug
                    debug(f"  Added doc ID={d.get('ID')} Title={d.get('Title')}")  ## debug

                session["_turn_docs"].extend(docs)

            # semantic_search / metadata_search list output
            elif isinstance(out, list):

                debug(f"Tool returned list with {len(out)} docs")        ## debug
                for d in out:                                            ## debug
                    debug(f"  Added doc ID={d.get('ID')} Title={d.get('Title')}")   ## debug
                
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

        debug("=== Memory Update Start ===")               ## debug
        debug(f"Assistant final turn text:\n{final_text}") ## debug

        # store assistant reply
        mem.add_turn("assistant", final_text)

        # store retrieved docs
        if session["_turn_docs"]:

            debug(f"Saving {len(session['_turn_docs'])} docs into memory")    ## debug
            for d in session["_turn_docs"]:                                   ## debug
                if isinstance(d, dict):                                       ## debug
                    debug(f"  Doc ID={d.get('ID')} Title={d.get('Title')}")   ## debug
                elif hasattr(d, "metadata"):                                  ## debug
                    debug(f"  Doc ID={d.metadata.get('ID')} Title={d.metadata.get('Title')}")  ## debug
                else:                                                         ## debug
                    debug(f"  Doc (unknown format): {d}")                     ## debug
                
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