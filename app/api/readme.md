# `app/api` endpoints

Auth = `Authorization: Bearer <Supabase access token>` unless noted.

HTTP paths are unchanged. Modules are split by dependency boundary:

- `public/` — papers profile; no Supabase auth, Chat, or agents
- `user/` — full-app auth/library/notes/resolution/export/config
- `chat.py` — full-app chat transport (application logic lives in `app/chat/`)
- `route_allowlist.py` — explicit papers vs full blueprint registration lists

## `public/health.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/health` | no | Profile name + capability flags. Full profile adds `agentRuntime` admission/metrics snapshot (`ready` vs `accepting`); papers returns `agentRuntime: null`. No user/run data. |

## `public/papers.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET`/`POST` | `/getPapers` | no | Query via args (GET) or JSON (POST). Params: `search_query`, `search_mode` (`exact`, `bm25`, or `vector`), optional `embedding_model`, `title`, `abstract`, `author`, `source`, `keyword`, year/citation ranges, `id_list`, `offset`, `limit` (max 100). |
| `POST` | `/getSimilarPapers` | no | RRF similarity from seed paper IDs. |
| `POST` | `/getPaperCitations` | no | OpenAlex references / cited-by. |

## `public/lookup.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/getPaperById` | no | Public corpus ID lookup. |
| `POST` | `/getPaperByTitle` | no | Title lookup. |

## `public/corpus.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/getUmapPoints` | no | Cached UMAP points for the map. |
| `GET` | `/getMetaData` | no | Filter facets; live Zilliz fallback if cache miss. |

## `user/config.py`

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/getPublicConfig` | none | Full-app browser settings: PDF size limit, `availableModels`, `defaultModel`. |

## `user/export.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `POST` | `/checkoutPapers` | no | BibTeX export for the full app workflow. |

## `user/papers.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `POST` | `/papers/resolve` | optional | Resolve public + authenticated library paper IDs. |

## `chat.py` (Flask) and ASGI `/chat/v2`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `POST` | `/chat/import` | required | Body `{ conversations: [...] }`. Idempotent guest→cloud import. |
| `GET` | `/chat/conversations` | required | User's cloud chat history. |
| `PUT` | `/chat/conversations/{id}/closed` | required | Body `{ is_closed: boolean }`; saves the tab visibility state. |
| `POST` | `/chat/v2` | optional | **ASGI-only** typed SSE (`text/event-stream`). Requires `client_request_id`. Backend assigns `agentRunId` / `assistantMessageId` in `run.started`. Message limit 10,000 chars. `agent_v2` owns talk, clarification, paper search, and selected-paper synthesis. |

Pre-stream failures on `/chat/v2` (validation, auth, unavailable executor) return JSON:

```json
{"detail": "client_request_id is required"}
```

with the usual HTTP status (`400` / `401` / `403` / `503`). After headers are sent, failures are terminal SSE `run.failed` events instead.

## `user/library.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/library/papers` | required | All `user_papers` (incl. `origin`). `?saved=true` → only `is_saved`. |
| `PUT` | `/library/papers/{paper_id}/saved` | required | JSON `Paper` metadata. Sets `is_saved=true`. |
| `DELETE` | `/library/papers/{paper_id}/saved` | required | Unsave. Corpus rows delete when no file; imported rows are retained. |
| `DELETE` | `/library/papers/{paper_id}` | required | Permanently delete a user-imported paper and its uploaded full text. |
| `POST` | `/library/papers/saved` | required | Body `{ papers: Paper[] }` (max 100). Bulk upsert as saved. |
| `POST` | `/library/papers/unsave` | required | Body `{ paper_ids: string[] }` (max 100). Bulk Unsave using each row's origin/file rule. |
| `POST` | `/library/papers/import` | required | Body `{ items: [{ paper, raw? }], also_save?: boolean }` (max 100). `paper.title` and `paper.abstract` are required. Each valid item is stored as an independent `origin=user` paper with a `user:` ID; `also_save` defaults to `true`; response has per-item `imported` / `invalid` results. |
| `PUT` | `/library/papers/{paper_id}/file` | required | multipart: `file` (PDF) + `metadata` (JSON `Paper`). |
| `DELETE` | `/library/papers/{paper_id}/file` | required | Deletes Azure file, clears upload fields; drops unsaved empty rows. |

## `user/notes.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/notes` | required | One research-notes document per user. Missing row → `200` with empty `content`. |
| `PUT` | `/notes` | required | Body `{ content: string }`. Upsert by `user_id`. |
