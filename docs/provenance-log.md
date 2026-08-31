# Provenance logging (backend)

All provenance events share a core envelope and land in the same GCP sink
(`extra={"provenance_event": True}`). Use the top-level **`source`** field to
filter by channel.

## Core envelope

```ts
{
  schemaVersion: 1,
  source: 'ui' | 'mcp' | 'agent',
  eventId: string,
  timestamp: number, // Unix ms
  action: string,
  eventData: Record<string, unknown>,
}
```

Additional fields depend on `source` (see below). Omit fields that do not apply;
do not use empty-string placeholders.

## UI events (`source: 'ui'`)

Transport: browser → Socket.IO `log_event` → `validate_ui_event()` →
`emit_provenance_event()`.

Required:

- `sessionId`
- `actorType`: `'user' | 'agent' | 'system'`
- `eventData`

Common optional fields: `userId`, `userIsLoggedIn`, `uiView`, `agentRunId`,
`clientRequestId`.

The frontend stamps `source: 'ui'` in `logProvenanceEvent()`. Older clients
without `source` are normalized to `'ui'` on ingest.

## MCP events (`source: 'mcp'`)

Transport: MCP tool wrapper on the server → `log_mcp_tool_event()` →
`emit_provenance_event()`.

Required:

- `action`: `tool.<tool_name>` (e.g. `tool.search_papers_bm25`)
- `eventData.tool`, `eventData.args`, `eventData.status`, `eventData.latencyMs`

Optional:

- `mcpSessionId` — when the MCP SDK exposes a session id
- `agentRunId` — external run id (benchmark / client metadata)
- `client` — e.g. `"cursor"`

Omit: `sessionId`, `actorType`, `uiView`, `userId`.

`eventData.args` is sanitized (truncated strings, bounded lists). Result payloads
are summarized (`resultTotal`, `resultCount`, `paperId`, …), not logged in full.

## Agent events (`source: 'agent'`)

Reserved for a future pass aligning built-in chat agent traces
(`SearchV2Trace`) with this envelope. Not emitted yet.

## Filtering (GCP)

Project (from service-account credentials): **`vitality-2-logging`**

Log name: **`projects/vitality-2-logging/logs/vitality2`**

Open [Logs Explorer](https://console.cloud.google.com/logs/query;query=) and select project
`vitality-2-logging`. Use a recent time window (e.g. Last 1 hour).

| Goal | Query |
|------|-------|
| Any vitality2 app log | `logName="projects/vitality-2-logging/logs/vitality2"` |
| All MCP tool calls | `logName="projects/vitality-2-logging/logs/vitality2" jsonPayload.source="mcp"` |
| One MCP tool | add `jsonPayload.action="tool.search_papers_bm25"` |
| All UI provenance | `jsonPayload.source="ui"` |
| Search by overview line | `textPayload:"MCP tool.search_papers_bm25"` or `textPayload:"Provenance ui"` |

Structured fields (`source`, `action`, `eventData`, …) are in **`jsonPayload`**.
The one-line overview is in **`textPayload`**. Allow 1–2 minutes for batch delivery.

Local verification: `python test_logging.py` (writes a probe and flushes handlers).

## TODO — agent events reuse `agentRunId`

The frontend already mints an `agentRunId` for UI actions that start an agent run
(e.g. `chat.submit`, `paper.find-similar`), puts it on the provenance envelope
(top-level, not inside `eventData`), and sends the same value to the API as
`agent_run_id` when applicable.

**Do:**

- Accept `agent_run_id` on chat / other agent-entry endpoints.
- Pass it into agent run logging (`agents/agent_v2/logging.py` and related traces)
  so every `source: "agent"` provenance event for that run uses the **same**
  `agentRunId`.
- Stop minting a separate id when the client already provided one
  (today `SearchV2Trace.create` falls back to `uuid4().hex` as `trace_id`).

**Prefer:** map client `agent_run_id` → provenance `agentRunId` (and align or alias
`trace_id` with it so analysis can join UI submit ↔ agent decisions).

Agent event catalogue (`action` / `eventData` for router, search, synthesis, etc.)
is still out of scope for the first pass; wire the shared id first.
