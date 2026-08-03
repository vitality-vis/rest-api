# Provenance logging (backend)

Shared event envelope (same shape as the frontend `ProvenanceEvent`):

```ts
{
  schemaVersion: 1,
  eventId: string,
  timestamp: number, // client Unix ms for UI events; server time OK for agent/system
  sessionId: string,
  actorType: 'user' | 'agent' | 'system',
  userId?: string,
  userIsLoggedIn?: boolean,
  uiView?: string,
  action: string,
  agentRunId?: string,
  eventData: Record<string, unknown>,
}
```

## TODO — agent events reuse `agentRunId`

The frontend already mints an `agentRunId` for UI actions that start an agent run
(e.g. `chat.submit`, `paper.find-similar`), puts it on the provenance envelope
(top-level, not inside `eventData`), and sends the same value to the API as
`agent_run_id` when applicable.

**Do:**

- Accept `agent_run_id` on chat / other agent-entry endpoints.
- Pass it into agent run logging (`agents/agent_v2/logging.py` and related traces)
  so every `actorType: "agent"` provenance event for that run uses the **same**
  `agentRunId`.
- Stop minting a separate id when the client already provided one
  (today `SearchV2Trace.create` falls back to `uuid4().hex` as `trace_id`).

**Prefer:** map client `agent_run_id` → provenance `agentRunId` (and align or alias
`trace_id` with it so analysis can join UI submit ↔ agent decisions).

Agent event catalogue (`action` / `eventData` for router, search, synthesis, etc.)
is still out of scope for the first pass; wire the shared id first.
