"""Agent layer for orchestration, mailboxes, workflow state, and worker agents."""

from .mailbox import Mailbox, AgentMessage
from .status import AgentStatus, is_final
from .registry import AgentRegistryBase, AgentRegistry
from .base_agent import BaseAgent
from .sub_agent_task import SubAgentTask, SubAgentResult
from .sub_agent_base import SubAgentBase
from .action import OrchestratorActionType, OrchestratorAction
from .chat import (
    WorkflowState,
    MainOrchestratorAgent,
    AssessorAgent,
    AssessmentResult,
    PlannerAgent,
    ExecutionPlan,
    PlanStep,
    ResearchTask,
    ResearchPlan,
    ResearchTaskType,
    ResearchPlannerAgent,
    CollectPapersAgent,
)

__all__ = [
    # Messaging
    "Mailbox", "AgentMessage",
    # Lifecycle
    "AgentStatus", "is_final",
    # Registry
    "AgentRegistryBase", "AgentRegistry",
    # Agent base classes
    "BaseAgent", "SubAgentBase",
    # Task contracts
    "SubAgentTask", "SubAgentResult",
    # Action model
    "OrchestratorActionType", "OrchestratorAction",
    # Workflow
    "WorkflowState",
    # Orchestrator
    "MainOrchestratorAgent",
    # Subagents
    "AssessorAgent", "AssessmentResult",
    "PlannerAgent", "ExecutionPlan", "PlanStep",
    # Research-task layer
    "ResearchTask", "ResearchPlan", "ResearchTaskType",
    "ResearchPlannerAgent", "CollectPapersAgent",
]
