"""Literature-review (LR) human-in-the-loop demo package.

A self-contained LangGraph demo proving the
plan -> human approval -> execute -> human review -> write loop.

It is intentionally isolated from the existing chat/RAG agents and uses
deterministic placeholder logic so it always runs without a real LLM or search
tool. Replace the node bodies with real agents/tools later.
"""
