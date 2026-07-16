"""
MainOrchestratorAgent: root agent that owns the per-turn execution pipeline.

Does NOT inherit BaseAgent — it is the root of the hierarchy, not a
subordinate worker. It drives subagents exclusively via call_subagent(),
which hides the execution model behind a clean async boundary.

Execution model
---------------
Explicit while-not-done loop:

    while not state.done:
        action = decide_next_action(state)
        async for chunk in execute_action(action, state):
            yield chunk
        update_after_action(action, state)

No LangChain AgentExecutor or astream_events(). The LLM is called directly
via llm.astream() for final-answer generation and llm.invoke() for sync steps
(fast-reply, small-talk, rewrite, classify, gateway).

Import note
-----------
Imports service.agents.agent_utils (not service.application.agent_service) to avoid a circular
import: agent_service → agents/__init__ → chat/orchestrator → agent_service ✗
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import AsyncGenerator, Optional, Tuple

from ..action import OrchestratorAction, OrchestratorActionType
from ..registry import AgentRegistryBase, AgentRegistry
from ..status import AgentStatus
from ..sub_agent_base import SubAgentBase
from ..sub_agent_task import SubAgentTask, SubAgentResult
from .workflow_state import WorkflowState

logger = logging.getLogger(__name__)

# Matches the [SIGNAL:STATUS:<label>] tokens the orchestrator streams to the
# chat window. Used to log every visible "thinking" step on the backend.
_STATUS_SIGNAL_RE = re.compile(r"\[SIGNAL:STATUS:([^\]]+)\]")

# ── Answer-generation system prompt ──────────────────────────────────────────
# Used by GENERATE_FINAL_ANSWER. Tool-selection instructions are omitted
# because tool selection is now deterministic (handled by decide_next_action).

_ANSWER_SYSTEM_PROMPT = """\
You are an intelligent research assistant.
Retrieved papers have been provided to you as context.
Your task: answer the user's question based on the retrieved information.

## FINAL ANSWER FORMAT

When returning papers from a search (semantic_search / metadata_search / mixed_search / load_more_papers):
1. Start with one or two natural sentences introducing the list.
2. Output ALL papers. Do NOT omit or reorder them.
3. Use this compact format:
   - **Title:** <title> [[ID:<paper id>]]
     Authors: <authors>
     Year: <year>
     Source: <source>
4. Do NOT include Abstract or Score unless the user explicitly asks.
5. If the context contains [SIGNAL:SHOW_LOAD_MORE], copy it EXACTLY at the end.

When answering a question (rag_semantic_qa):
- Provide a clear, concise explanation or analysis.
- Cite papers as: *Paper Title* [[ID:<paper id>]]
- Only cite papers that are clearly relevant to the question.

## SAFETY RULES
- Never invent IDs, authors, years, sources, or titles.
- Only reference papers explicitly present in the retrieved context.
- If no relevant papers were retrieved, say so clearly.
"""


class MainOrchestratorAgent:
    """
    Root orchestrator — owns the full per-turn pipeline.

    Boundaries
    ----------
    Owns   : WorkflowState lifecycle, subagent dispatch, pipeline stages
    Injects: session["llm"], session["tool_executor"], session["registry"]
    Never  : calls agent.run_from_mailbox() directly
    Always : uses call_subagent() for all subagent work
    """

    def __init__(self, session: dict, chat_id: str) -> None:
        self._session:  dict              = session
        self._chat_id:  str               = chat_id
        self._registry: AgentRegistryBase = session["registry"]
        self._llm                         = session["llm"]
        self._mem                         = session["mem"]

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self, state: WorkflowState) -> AsyncGenerator[str, None]:
        """Execute a full orchestrator turn. Yields streaming text and [SIGNAL:...] tokens."""
        await self._initialize_state(state)

        # Per-turn step tracking for backend logging of every chat-window step.
        turn_start = time.perf_counter()
        step_counter = {"n": 0}
        logger.info(
            "[ChatStep] chat_id=%s ▶ turn start | input=%r",
            self._chat_id, (state.user_input or "").strip()[:200],
        )

        while not state.done:
            action = self.decide_next_action(state)
            state.current_action = action
            state.iteration += 1
            logger.info(
                "[Orchestrator] iter=%d action=%s tool=%s reason=%r",
                state.iteration, action.type.value, action.tool_name or "-", action.reason,
            )
            async for chunk in self.execute_action(action, state):
                # Log every status step shown in the chat window as it is emitted.
                if isinstance(chunk, str) and "[SIGNAL:STATUS:" in chunk:
                    for label in _STATUS_SIGNAL_RE.findall(chunk):
                        step_counter["n"] += 1
                        logger.info(
                            "[ChatStep] chat_id=%s step #%d: %s (+%.2fs)",
                            self._chat_id, step_counter["n"], label,
                            time.perf_counter() - turn_start,
                        )
                yield chunk
            self.update_after_action(action, state)

        logger.info(
            "[ChatStep] chat_id=%s ■ turn done | %d step(s) in %.2fs",
            self._chat_id, step_counter["n"], time.perf_counter() - turn_start,
        )

    # ── State initialization ──────────────────────────────────────────────────

    async def _initialize_state(self, state: WorkflowState) -> None:
        """Trim clean_input and configure self-RAG defaults."""
        if not state.clean_input:
            state.clean_input = state.user_input.strip()[:2000]
        if not state.agent_input:
            state.agent_input = state.clean_input

    # ── Action decision ───────────────────────────────────────────────────────

    def decide_next_action(self, state: WorkflowState) -> OrchestratorAction:
        """
        Pure function: maps current WorkflowState to the next action.
        All mutations happen in execute_action / update_after_action.
        """
        from service.application.pipeline import Intent

        # ── Error threshold ────────────────────────────────────────────────
        if len(state.errors) >= 3:
            return OrchestratorAction(
                type=OrchestratorActionType.ERROR_FALLBACK,
                reason="too many errors",
            )

        # ── Gateway signaled (clarification requested) ─────────────────────
        if state._gateway_signaled:
            return OrchestratorAction(type=OrchestratorActionType.FINALIZE)

        # ── Selected papers: skip L2/L3/L4 preprocessing ──────────────────
        if state.has_selected_papers():
            if not state._gateway_done:
                return OrchestratorAction(
                    type=OrchestratorActionType.QUERY_GATEWAY,
                    reason="selected_papers",
                )
            # Papers already resolved into agent_input → generate answer
            if not state.final_answer:
                return OrchestratorAction(type=OrchestratorActionType.GENERATE_FINAL_ANSWER)
            return OrchestratorAction(type=OrchestratorActionType.FINALIZE)

        # ── L2: Rewrite query ──────────────────────────────────────────────
        if not state._rewrite_done:
            return OrchestratorAction(type=OrchestratorActionType.REWRITE_QUERY)

        # If fast-reply was triggered during rewrite, we're done
        if state.done:
            return OrchestratorAction(type=OrchestratorActionType.FINALIZE)

        # ── L3: Classify intent ────────────────────────────────────────────
        if state.intent is None:
            return OrchestratorAction(type=OrchestratorActionType.CLASSIFY_INTENT)

        # ── Small talk: direct LLM reply, no retrieval ────────────────────
        if state.intent.intent == Intent.SMALL_TALK:
            if not state.final_answer:
                return OrchestratorAction(
                    type=OrchestratorActionType.GENERATE_FINAL_ANSWER,
                    reason="small_talk",
                )
            return OrchestratorAction(type=OrchestratorActionType.FINALIZE)

        # ── L4.x: Query gateway ────────────────────────────────────────────
        if not state._gateway_done:
            return OrchestratorAction(type=OrchestratorActionType.QUERY_GATEWAY)

        # ── Tool execution ─────────────────────────────────────────────────
        # First call: no outputs yet
        if not state.tool_outputs:
            # Research-task path (Planner → ResearchTask → Agent). Only the
            # FIRST retrieval is routed through here; everything downstream
            # (self-RAG, answer generation) is unchanged. Falls back to the
            # deterministic intent→tool path on any planning failure.
            if self._research_mode_active(state):
                if state.research_plan is None:
                    return OrchestratorAction(
                        type=OrchestratorActionType.PLAN_RESEARCH,
                    )
                runnable = (
                    getattr(state.research_plan, "tasks", None)
                    and not getattr(state.research_plan, "errors", None)
                )
                if runnable and not state.research_plan_executed:
                    return OrchestratorAction(
                        type=OrchestratorActionType.EXECUTE_RESEARCH_PLAN,
                    )
                # plan empty/failed/executed-without-output → deterministic fallback

            tool_name, tool_args = self._select_tool(state)
            return OrchestratorAction(
                type=OrchestratorActionType.EXECUTE_TOOL,
                tool_name=tool_name,
                tool_args=tool_args,
            )

        # Subsequent call after self-RAG refinement
        if state.rag_iter > 0 and len(state.tool_outputs) <= state.rag_iter:
            tool_name, tool_args = self._select_refined_tool(state)
            return OrchestratorAction(
                type=OrchestratorActionType.EXECUTE_TOOL,
                tool_name=tool_name,
                tool_args=tool_args,
                reason="self_rag_refinement",
            )

        # ── Self-RAG assessment (iter 0 only) ──────────────────────────────
        if state.use_self_rag and state.rag_iter == 0 and state.assessment is None:
            return OrchestratorAction(type=OrchestratorActionType.ASSESS_RETRIEVAL)

        # ── Refinement (when assessment is insufficient AND query needs broad coverage)
        # Definition questions ("what is MCTS?") skip refinement even when the
        # assessment score is low — re-retrieval won't improve a narrow factual answer.
        if (
            state.use_self_rag
            and state.assessment is not None
            and not state.assessment.get("is_sufficient")
            and state.assessment.get("refined_query")
            and state.rag_iter < state.max_rag_iter
            and self._requires_broad_coverage(state)
        ):
            return OrchestratorAction(
                type=OrchestratorActionType.REFINE_QUERY,
                reason=state.assessment["refined_query"],
            )

        # ── Generate final answer ──────────────────────────────────────────
        if not state.final_answer:
            return OrchestratorAction(type=OrchestratorActionType.GENERATE_FINAL_ANSWER)

        return OrchestratorAction(type=OrchestratorActionType.FINALIZE)

    # ── Action dispatch ───────────────────────────────────────────────────────

    async def execute_action(
        self, action: OrchestratorAction, state: WorkflowState
    ) -> AsyncGenerator[str, None]:
        """Dispatch to the appropriate handler. Yields streaming chunks."""
        t = action.type
        if t == OrchestratorActionType.REWRITE_QUERY:
            async for chunk in self._exec_rewrite(state):
                yield chunk
        elif t == OrchestratorActionType.CLASSIFY_INTENT:
            async for chunk in self._exec_classify(state):
                yield chunk
        elif t == OrchestratorActionType.QUERY_GATEWAY:
            async for chunk in self._exec_gateway(action, state):
                yield chunk
        elif t == OrchestratorActionType.PLAN_RESEARCH:
            async for chunk in self._exec_plan_research(state):
                yield chunk
        elif t == OrchestratorActionType.EXECUTE_RESEARCH_PLAN:
            async for chunk in self._exec_research_plan(state):
                yield chunk
        elif t == OrchestratorActionType.EXECUTE_TOOL:
            async for chunk in self._exec_tool(action, state):
                yield chunk
        elif t == OrchestratorActionType.ASSESS_RETRIEVAL:
            async for chunk in self._exec_assess(state):
                yield chunk
        elif t == OrchestratorActionType.REFINE_QUERY:
            async for chunk in self._exec_refine(action, state):
                yield chunk
        elif t == OrchestratorActionType.GENERATE_FINAL_ANSWER:
            async for chunk in self._exec_generate_answer(action, state):
                yield chunk
        elif t == OrchestratorActionType.FINALIZE:
            async for chunk in self._exec_finalize(state):
                yield chunk
        elif t == OrchestratorActionType.ERROR_FALLBACK:
            async for chunk in self._exec_error_fallback(state):
                yield chunk
        # FAST_REPLY is handled inside REWRITE_QUERY

    def update_after_action(
        self, action: OrchestratorAction, state: WorkflowState
    ) -> None:
        """Post-action state mutations (non-yielding side-effects)."""
        t = action.type
        if t == OrchestratorActionType.FINALIZE:
            state.done = True
        elif t == OrchestratorActionType.ERROR_FALLBACK:
            state.done = True
        elif t == OrchestratorActionType.EXECUTE_RESEARCH_PLAN:
            # Mark executed so we don't loop the plan; if it produced no output,
            # decide_next_action falls back to the deterministic tool path.
            state.research_plan_executed = True
        elif t == OrchestratorActionType.REFINE_QUERY:
            state.rag_iter += 1
            state.refined_query = action.reason
            # Clear assessment so decide_next_action won't assess again after refinement
            state.assessment = None
        elif t == OrchestratorActionType.GENERATE_FINAL_ANSWER:
            if action.reason == "small_talk":
                # Schedule memory update for non-retrieval answers
                asyncio.create_task(
                    _async_update_memory(self._session, state.final_answer)
                )

    # ── Action handlers ───────────────────────────────────────────────────────

    async def _exec_rewrite(self, state: WorkflowState) -> AsyncGenerator[str, None]:
        """L0 fast-router + L2 rewrite."""
        from langchain.schema import HumanMessage
        from service.application.pipeline import FastRouter, rewrite_query
        from service.agents.agent_utils import _fallback_direct_answer

        clean_input = state.user_input.strip()[:2000]
        state.clean_input = clean_input

        # L0: Zero-LLM fast path
        fast_result = FastRouter().check(clean_input)
        if fast_result:
            logger.info("[L0] fast-route: %s — skipping L2/L3", fast_result.intent)
            yield "[SIGNAL:STATUS:Thinking]"
            try:
                response = self._llm.invoke([HumanMessage(content=clean_input)]).content
                state.final_answer = response
                yield response
                self._mem.add_turn("assistant", response)
            except Exception as exc:
                logger.warning("[L0] fast-route LLM failed: %s", exc)
                async for chunk in _fallback_direct_answer(
                    self._llm, clean_input, mem=self._mem
                ):
                    state.final_answer += chunk
                    yield chunk
            state.done = True
            state._rewrite_done = True
            return

        yield "[SIGNAL:STATUS:Analyzing your query]"

        # L2: Rewrite against conversation history
        try:
            rw = rewrite_query(
                clean_input, self._llm,
                session=self._session, chat_id=state.chat_id,
            )
            state.clean_input = rw.rewritten_query
            logger.info("[L2] rewrite=%s type=%s", rw.was_rewritten, rw.rewrite_type)
        except Exception as exc:
            logger.warning("[L2] rewrite failed: %s", exc)

        state._rewrite_done = True

    async def _exec_classify(self, state: WorkflowState) -> AsyncGenerator[str, None]:
        """L3: Classify intent and extract slots."""
        from service.application.pipeline import classify_intent

        try:
            intent = classify_intent(
                state.clean_input, self._llm,
                session=self._session, chat_id=state.chat_id,
            )
            state.intent = intent
            logger.info(
                "[L3] intent=%s conf=%.2f hint=%s",
                intent.intent, intent.confidence, intent.tool_hint,
            )
        except Exception as exc:
            logger.warning("[L3] classify failed: %s", exc)
        return
        yield  # make async generator

    async def _exec_gateway(
        self, action: OrchestratorAction, state: WorkflowState
    ) -> AsyncGenerator[str, None]:
        """L4.x: Query understanding gate + selected-papers resolution."""
        from service.application.pipeline import GatewayAction, GatewayResult, QueryGateway
        from service.agents.agent_utils import _resolve_selected_papers_from_cache

        # ── Selected papers bypass ─────────────────────────────────────────
        if action.reason == "selected_papers":
            try:
                ids    = list(state.selected_paper_ids)
                titles = list(state.selected_paper_titles)
                n      = max(len(ids), len(titles))
                requested = [
                    (titles[i] if i < len(titles) else None,
                     ids[i]    if i < len(ids)    else None)
                    for i in range(n)
                ]
                if requested:
                    agent_input, _ = _resolve_selected_papers_from_cache(
                        state.chat_id, requested, state.user_input.strip()
                    )
                    state.agent_input = agent_input
            except Exception as exc:
                logger.warning("[Gateway] resolve_selected_papers failed: %s", exc)
                state.agent_input = state.user_input
            state._gateway_done = True
            return
            yield

        # ── Standard gateway ───────────────────────────────────────────────
        skip_gateway = self._session.pop("_skip_gateway_next", False)
        if skip_gateway:
            from service.application.pipeline import QueryGateway as _QG
            gw_result = GatewayResult(
                action=GatewayAction.PROCEED,
                agent_input=(
                    _QG._build_agent_input(state.intent, state.clean_input)
                    if state.intent else state.clean_input
                ),
            )
        else:
            try:
                gw_result = QueryGateway().process(
                    state.intent, state.clean_input, self._session
                )
            except Exception as exc:
                logger.warning("[Gateway] failed: %s", exc)
                gw_result = None

            if gw_result and gw_result.action == GatewayAction.SIGNAL:
                self._session["_skip_gateway_next"] = True
                state._gateway_signaled = True
                self._mem.add_turn("assistant", gw_result.signal)
                yield gw_result.signal
                state._gateway_done = True
                return

        if state.intent:
            self._session["last_intent"] = state.intent

        state.agent_input = (
            gw_result.agent_input
            if gw_result and gw_result.agent_input
            else state.clean_input
        )

        # ── Configure self-RAG (RAG_QA only) ──────────────────────────────
        from service.application.pipeline import Intent
        state.use_self_rag = (
            state.intent is not None
            and state.intent.intent == Intent.RAG_QA
            and self._session.get("self_rag_enabled", True)
        )
        state.max_rag_iter = (
            self._session.get("self_rag_max_iter", 2) if state.use_self_rag else 1
        )

        state._gateway_done = True

    # ── Research-task path handlers ────────────────────────────────────────────

    def _research_mode_active(self, state: WorkflowState) -> bool:
        """
        True when the Planner→ResearchTask path should handle the first
        retrieval. Gated by a session flag (default off) and restricted to
        retrieval-style intents so SMALL_TALK / LOAD_MORE keep their existing
        deterministic behavior.
        """
        from service.application.pipeline import Intent

        if not self._session.get("research_planning_enabled", False):
            return False
        if state.intent is None:
            return False
        if state.intent.intent in (Intent.SMALL_TALK, Intent.LOAD_MORE):
            return False
        return True

    async def _exec_plan_research(
        self, state: WorkflowState
    ) -> AsyncGenerator[str, None]:
        """Ask the ResearchPlannerAgent for a ResearchPlan (Planner → tasks)."""
        from .research_task import ResearchPlan, ResearchTask, ResearchTaskType

        yield "[SIGNAL:STATUS:Planning research]"

        task = SubAgentTask(
            agent_path=AgentRegistry.PLANNER,
            payload={
                "user_request": state.agent_input or state.clean_input,
                "reply_to":     AgentRegistry.ROOT,
            },
        )
        task.payload["task_id"] = task.task_id

        result = await self.call_subagent(AgentRegistry.PLANNER, task)

        plan = ResearchPlan()
        if result.is_ok() and result.output.get("type") == "research_plan":
            errors = list(result.output.get("errors") or [])
            tasks: list = []
            for raw in (result.output.get("tasks") or []):
                try:
                    ttype = ResearchTaskType.from_value(raw.get("task"))
                except ValueError:
                    errors.append(f"unknown task '{raw.get('task')}'")
                    continue
                tasks.append(ResearchTask(type=ttype, params=raw.get("params") or {}))
            plan = ResearchPlan(tasks=tasks, errors=errors)
        else:
            plan = ResearchPlan(errors=[result.error or "planner returned no plan"])

        if plan.errors:
            logger.info("[Orchestrator] research plan unusable (%s) → deterministic fallback",
                        plan.errors)
        else:
            logger.info("[Orchestrator] research plan: [%s]",
                        ", ".join(t.type.value for t in plan.tasks))
        state.research_plan = plan

    async def _exec_research_plan(
        self, state: WorkflowState
    ) -> AsyncGenerator[str, None]:
        """Run the ResearchPlan via TaskExecutor and fold results into state."""
        from .task_executor import TaskExecutor

        yield "[SIGNAL:STATUS:Collecting papers]"

        executor = TaskExecutor(self.call_subagent, self._session)
        results = await executor.execute_plan(state.research_plan, state)

        for r in results:
            if r.get("status") == "ok" and r.get("output"):
                state.tool_outputs.append(r["output"])
                state.retrieved_docs = r.get("docs") or []
                self._session["_turn_docs"] = list(r.get("docs") or [])
            elif r.get("error"):
                state.errors.append(f"research task '{r.get('task')}' failed: {r['error']}")

    async def _exec_tool(
        self, action: OrchestratorAction, state: WorkflowState
    ) -> AsyncGenerator[str, None]:
        """Execute the selected tool and update retrieved_docs + tool_outputs."""
        tool_executor = self._session.get("tool_executor")
        if tool_executor is None:
            state.errors.append("tool_executor missing from session")
            return
            yield

        status_msg = {
            "semantic_search":  "Searching papers semantically",
            "metadata_search":  "Searching paper metadata",
            "rag_semantic_qa":  "Analyzing retrieved papers",
            "load_more_papers": "Loading more results",
            "mixed_search":     "Running mixed search",
        }.get(action.tool_name or "", "Searching")
        yield f"[SIGNAL:STATUS:{status_msg}]"

        result = tool_executor.execute(action.tool_name or "", action.tool_args)

        if not result.is_ok():
            err_msg = f"Tool {action.tool_name} failed: {result.error}"
            logger.warning("[Orchestrator] %s", err_msg)
            state.errors.append(err_msg)
            return

        state.tool_outputs.append(result.output)
        state.retrieved_docs = result.docs
        # Keep _turn_docs in sync for memory manager
        self._session["_turn_docs"] = list(result.docs)

        # load_more_papers output is ready-to-stream (already formatted list)
        if action.tool_name == "load_more_papers" and result.output.strip():
            state.final_answer = result.output
            yield result.output

    async def _exec_assess(self, state: WorkflowState) -> AsyncGenerator[str, None]:
        """Dispatch retrieval quality assessment to AssessorAgent."""
        assessment = await self._assess_retrieval(state)
        if assessment:
            state.assessment = assessment
            logger.info(
                "[self-RAG] score=%.2f sufficient=%s",
                assessment.get("score", 0), assessment.get("is_sufficient"),
            )
        else:
            # No docs or assessor unavailable → treat as sufficient
            state.assessment = {"is_sufficient": True, "score": 0.5}
        return
        yield

    async def _exec_refine(
        self, action: OrchestratorAction, state: WorkflowState
    ) -> AsyncGenerator[str, None]:
        """Emit a frontend status signal that search is being refined."""
        logger.info(
            "[self-RAG] iter=%d score=%.2f insufficient → refining: %r",
            state.rag_iter,
            state.assessment.get("score", 0) if state.assessment else 0,
            action.reason,
        )
        yield "[SIGNAL:STATUS:Refining search for better coverage]"

    async def _exec_generate_answer(
        self, action: OrchestratorAction, state: WorkflowState
    ) -> AsyncGenerator[str, None]:
        """Generate the final answer by calling the LLM with tool output as context."""
        from langchain.schema import HumanMessage, SystemMessage
        from service.agents.agent_utils import _fallback_direct_answer

        yield "[SIGNAL:STATUS:Generating answer]"

        # Small talk: direct LLM call, no tool context
        if action.reason == "small_talk":
            try:
                response = self._llm.invoke(
                    [HumanMessage(content=state.clean_input)]
                ).content
                state.final_answer = response
                self._mem.add_turn("assistant", response)
                yield response
            except Exception as exc:
                logger.warning("[Orchestrator] small_talk LLM failed: %s", exc)
                async for chunk in _fallback_direct_answer(
                    self._llm, state.clean_input, mem=self._mem
                ):
                    state.final_answer += chunk
                    yield chunk
            return

        # Build context from tool output
        tool_output = state.tool_outputs[-1] if state.tool_outputs else ""
        user_question = state.clean_input or state.user_input
        history = self._mem.get_history_text()

        # When assessment is insufficient but we are NOT refining (e.g. simple
        # definition question), tell the LLM that evidence may be limited so it
        # can add a brief caveat rather than silently generating a weak answer.
        limited_evidence = (
            state.use_self_rag
            and state.assessment is not None
            and not state.assessment.get("is_sufficient")
        )
        caveat_note = (
            "\n\nNote: The retrieved papers may not fully cover this specific topic. "
            "Answer from the available evidence; if it is insufficient, give a concise "
            "answer from general knowledge and note the limitation briefly."
            if limited_evidence else ""
        )

        if tool_output:
            user_content = (
                f"Retrieved information:\n{tool_output}\n\n"
                f"User question: {user_question}{caveat_note}"
            )
        else:
            user_content = user_question + caveat_note

        if history:
            user_content = f"Conversation history:\n{history}\n\n{user_content}"

        messages = [
            SystemMessage(content=_ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        full: list[str] = []
        try:
            async for chunk in self._llm.astream(messages):
                content = getattr(chunk, "content", None)
                if content and not getattr(chunk, "tool_call_chunks", None):
                    full.append(content)
                    yield content
        except Exception as exc:
            logger.warning("[Orchestrator] answer stream failed: %s", exc)
            async for chunk in _fallback_direct_answer(
                self._llm, user_question, mem=self._mem
            ):
                full.append(chunk)
                yield chunk

        state.final_answer = "".join(full)

    async def _exec_finalize(self, state: WorkflowState) -> AsyncGenerator[str, None]:
        """Persist memory and mark turn complete."""
        asyncio.create_task(_async_update_memory(self._session, state.final_answer))
        return
        yield

    async def _exec_error_fallback(
        self, state: WorkflowState
    ) -> AsyncGenerator[str, None]:
        """Stream a fallback answer when the pipeline accumulated too many errors."""
        from service.agents.agent_utils import _fallback_direct_answer

        logger.warning(
            "[Orchestrator] error fallback after %d errors: %s",
            len(state.errors), state.errors,
        )
        async for chunk in _fallback_direct_answer(
            self._llm, state.user_input, mem=self._mem
        ):
            state.final_answer += chunk
            yield chunk

    # ── Tool selection ────────────────────────────────────────────────────────

    def _select_tool(self, state: WorkflowState) -> Tuple[str, dict]:
        """
        Deterministic tool selection from intent + slots (first call).

        Delegates to the shared select_search_tool() helper so the orchestrator
        and CollectPapersAgent pick tools identically (single source of truth).
        """
        from .search_tool_selection import select_search_tool

        query = state.agent_input or state.clean_input
        return select_search_tool(state.intent, query, state.clean_input)

    def _select_refined_tool(self, state: WorkflowState) -> Tuple[str, dict]:
        """Tool selection for self-RAG refinement iteration."""
        refined = state.refined_query or state.clean_input
        return "rag_semantic_qa", {
            "query": refined,
            "question": state.clean_input,
        }

    @staticmethod
    def _requires_broad_coverage(state: WorkflowState) -> bool:
        """
        Returns True when the query benefits from broad re-retrieval (survey,
        overview, comparative, methods-level question). Returns False for simple
        definitional queries like "what is MCTS?" where re-retrieval adds no value.

        Used to gate REFINE_QUERY: a low assessment score on a definition question
        does not justify another retrieval round.
        """
        import re as _re

        query = (state.clean_input or state.user_input).lower().strip()

        # Explicit broad-coverage signals → always worth refining
        _BROAD = _re.compile(
            r"\b("
            r"survey|overview|review|"
            r"compare|comparison|versus|vs\.?|"
            r"methods?|approaches?|techniques?|algorithms?|frameworks?|"
            r"state.of.the.art|related.work|"
            r"recent\s+(?:work|papers?|advances?|developments?|research)|"
            r"literature|"
            r"how\s+(?:do|does|can|are|is)\b|"
            r"what\s+(?:are\s+the\s+)?(?:different\s+)?(?:methods?|approaches?|ways?)"
            r")\b",
            _re.I,
        )
        if _BROAD.search(query):
            return True

        # Short definitional patterns → narrow factual question, skip refinement
        _DEFN = _re.compile(
            r"^(?:what\s+is|what\s+are|define|explain|describe|tell\s+me\s+about)\b",
            _re.I,
        )
        if _DEFN.match(query) and len(query.split()) <= 10:
            return False

        # Default: assume broad coverage (allows refinement — safe fallback)
        return True

    # ── Subagent dispatch ─────────────────────────────────────────────────────

    async def call_subagent(
        self,
        agent_path: str,
        task: SubAgentTask,
    ) -> SubAgentResult:
        """
        Dispatch a task to a registered subagent and await its result.

        Current model: synchronous (agent.execute_task() runs inline).
        Future: replace body with asyncio.create_task pool dispatch.
        """
        agent = self._registry.get_agent(agent_path)
        if agent is None:
            logger.warning("[Orchestrator] call_subagent: no agent at %s", agent_path)
            return SubAgentResult(
                task_id=task.task_id,
                agent_path=agent_path,
                output={},
                status=AgentStatus.ERRORED,
                duration_ms=0.0,
                error=f"Agent not registered: {agent_path}",
            )

        if not isinstance(agent, SubAgentBase):
            logger.warning(
                "[Orchestrator] call_subagent: %s does not implement SubAgentBase",
                agent_path,
            )
            return SubAgentResult(
                task_id=task.task_id,
                agent_path=agent_path,
                output={},
                status=AgentStatus.ERRORED,
                duration_ms=0.0,
                error=f"Agent at {agent_path} is not a SubAgentBase",
            )

        return await agent.execute_task(task)

    # ── Self-RAG assessment dispatch ──────────────────────────────────────────

    async def _assess_retrieval(self, state: WorkflowState) -> Optional[dict]:
        """
        Dispatch retrieval quality assessment to AssessorAgent via call_subagent().
        Never calls assessor.run_from_mailbox() directly.
        """
        from service.application.retrieval_service import format_docs

        docs = state.retrieved_docs
        if not docs:
            return None

        try:
            formatted = format_docs(
                docs[:5], include_abstract=True, include_score=False
            )
            task = SubAgentTask(
                agent_path=AgentRegistry.ASSESSOR,
                payload={
                    "query":    state.clean_input or state.user_input,
                    "docs":     formatted,
                    "reply_to": AgentRegistry.ROOT,
                },
            )
            result = await self.call_subagent(AgentRegistry.ASSESSOR, task)
            if result.is_ok() and result.output.get("type") == "assessment_result":
                return result.output

        except Exception as exc:
            logger.warning("[self-RAG] assessment failed: %s — skipping", exc)

        return None


# ── Module-level helpers ──────────────────────────────────────────────────────

async def _async_update_memory(session: dict, final_text: str) -> None:
    """Thin wrapper so the import of async_update_memory stays in agent_utils."""
    from service.agents.agent_utils import async_update_memory
    await async_update_memory(session, final_text)
