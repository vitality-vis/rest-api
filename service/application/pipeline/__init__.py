"""
Pipeline package — all query-understanding stages in order.

┌─────────────────────────────────────────────────────────────┐
│  L0   FastRouter        fast_router.py   (zero LLM)         │
│  L2   rewrite_query     rewriter.py      (LLM)              │
│  L3   classify_intent   classifier.py    (LLM)              │
│  L4x  QueryGateway      gateway.py       (heuristic+signal) │
│       ToolApprovalGate  approval.py      (sync gate)        │
└─────────────────────────────────────────────────────────────┘

Usage in application/agent_service.py:
    from service.application.pipeline import (
        FastRouter,
        rewrite_query,
        Intent, classify_intent,
        GatewayAction, QueryGateway,
        ApprovalDecision, ToolApprovalGate,
    )
"""
from .fast_router import FastRouter, FastRouteResult
from .rewriter    import RewriteResult, rewrite_query
from .classifier  import Intent, ExtractedSlots, IntentResult, classify_intent
from .gateway     import GatewayAction, GatewayResult, QueryGateway
from .approval    import ApprovalDecision, ApprovalResult, ToolApprovalGate

__all__ = [
    # L0
    "FastRouter", "FastRouteResult",
    # L2
    "RewriteResult", "rewrite_query",
    # L3
    "Intent", "ExtractedSlots", "IntentResult", "classify_intent",
    # L4x
    "GatewayAction", "GatewayResult", "QueryGateway",
    # Tool gate
    "ApprovalDecision", "ApprovalResult", "ToolApprovalGate",
]
