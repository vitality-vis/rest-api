from dotenv import load_dotenv
load_dotenv()
import os
import re
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from service.memory.chat_memory import MemoryManager
from service.memory.session_state import SESSIONS
from service.agents.agent_utils import _fallback_direct_answer, async_update_memory
from service.agents import (
    AgentRegistry,
    AssessorAgent,
    CollectPapersAgent,
    MainOrchestratorAgent,
    ResearchPlannerAgent,
    WorkflowState,
)
from service.tools import ToolExecutor
from service.application.pipeline import ToolApprovalGate
import asyncio
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


def get_azure_llm():
    return NoStopAzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=1,  # GPT-5 only supports temperature=1
    )


def get_or_create_chat_session(chat_id: str) -> dict:
    """Create or return a per-chat session with its own LLM, registry, and tools."""

    if chat_id in SESSIONS:
        return SESSIONS[chat_id]

    logging.info("[AgentRunner] Creating new chat session for chat_id=%s", chat_id)

    # ── 1. LLM (one per session) ──────────────────────────────────────────────
    llm = get_azure_llm()

    # ── 2. Tool approval gate ─────────────────────────────────────────────────
    gate = ToolApprovalGate()

    # ── 3. Multi-agent registry ───────────────────────────────────────────────
    registry = AgentRegistry(chat_id)
    registry.register_mailbox(AgentRegistry.ROOT)
    assessor = AssessorAgent(chat_id=chat_id, registry=registry, llm=llm)
    registry.register(AgentRegistry.ASSESSOR, assessor)

    # ── 4. ToolExecutor (all tool calls go through this) ─────────────────────
    tool_executor = ToolExecutor(gate=gate, chat_id=chat_id)

    # ── 4b. Research-task agents (Planner → ResearchTask → Agent) ────────────
    # Only CollectPapersAgent is wired today; it reuses the ToolExecutor above.
    registry.register(
        AgentRegistry.PLANNER,
        ResearchPlannerAgent(chat_id=chat_id, registry=registry, llm=llm),
    )
    registry.register(
        AgentRegistry.TASK_COLLECT_PAPERS,
        CollectPapersAgent(chat_id=chat_id, registry=registry, tool_executor=tool_executor),
    )

    # Feature flag: route the first retrieval through the research-task path.
    # Defaults to the env var RESEARCH_PLANNING_ENABLED (default off → existing
    # deterministic intent→tool pipeline is byte-for-byte unchanged).
    research_enabled = os.getenv("RESEARCH_PLANNING_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )

    # ── 5. Store session ──────────────────────────────────────────────────────
    SESSIONS[chat_id] = {
        "llm":           llm,
        "mem":           MemoryManager(),
        "_turn_docs":    [],
        "registry":      registry,
        "approval_gate": gate,
        "tool_executor": tool_executor,
        "research_planning_enabled": research_enabled,
    }

    return SESSIONS[chat_id]


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


def convert_history_text_to_messages(history_text: str):
    messages = []
    for line in history_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("USER:"):
            messages.append(HumanMessage(content=line[len("USER:"):].strip()))
        elif line.startswith("ASSISTANT:"):
            messages.append(AIMessage(content=line[len("ASSISTANT:"):].strip()))
        else:
            messages.append(HumanMessage(content=line))
    return messages


# ── Public API ────────────────────────────────────────────────────────────────

def reset_session(chat_id: str) -> None:
    """Clear conversation and docs for a single chat tab."""
    if chat_id not in SESSIONS:
        return
    sess = SESSIONS[chat_id]
    if "mem" in sess:
        sess["mem"].clear()
    else:
        sess["mem"] = MemoryManager()
    sess["_turn_docs"] = []
    logging.info("[Session] Cleared chat session %s", chat_id)


def reset_all_sessions() -> None:
    """Clear all chat sessions completely."""
    SESSIONS.clear()
    logging.info("[Session] All sessions cleared.")


async def run_two_stage_rag_stream(
    user_input: str,
    chat_id: str = "default",
    selected_paper_ids: list = None,
    selected_paper_titles: list = None,
):
    """
    Primary streaming entry point for the research assistant.

    Creates (or reuses) a chat session, builds a WorkflowState for this turn,
    and delegates the full pipeline to MainOrchestratorAgent.run().

    Public contract:
      - Async generator yielding str chunks and [SIGNAL:...] tokens
      - selected_paper_ids / selected_paper_titles bypass preprocessing
      - Streaming signals, memory updates, self-RAG all handled by orchestrator
    """
    session = None
    try:
        session = get_or_create_chat_session(chat_id)
    except Exception as exc:
        logging.warning("[run_two_stage_rag_stream] session creation failed: %s", exc)
        llm = get_azure_llm()
        async for chunk in _fallback_direct_answer(llm, user_input):
            yield chunk
        return

    mem = session["mem"]
    session["_turn_docs"] = []
    mem.add_turn("user", user_input)

    state = WorkflowState(
        chat_id=chat_id,
        user_input=user_input,
        selected_paper_ids=list(selected_paper_ids or []),
        selected_paper_titles=list(selected_paper_titles or []),
        clean_input=user_input.strip()[:2000],
        agent_input=user_input,
    )

    orchestrator = MainOrchestratorAgent(session=session, chat_id=chat_id)
    async for chunk in orchestrator.run(state):
        yield chunk


__all__ = [
    "run_two_stage_rag_stream",
    "reset_session",
    "reset_all_sessions",
]
