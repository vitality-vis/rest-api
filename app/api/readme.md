# `app/api` endpoints

Auth = `Authorization: Bearer <Supabase access token>` unless noted.

## `bootstrap.py`

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/getPublicConfig` | none | Public, non-sensitive browser runtime settings such as the PDF size limit. |

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/getUmapPoints` | no | Cached UMAP points for the map. |
| `GET` | `/getMetaData` | no | Filter facets; live Zilliz fallback if cache miss. |

## `papers.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET`/`POST` | `/getPapers` | no | Query via args (GET) or JSON (POST). Params: `search_query`, `title`, `abstract`, `author`, `source`, `keyword`, year/citation ranges, `id_list`, `offset`, `limit` (max 100). |

## `chat.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `POST` | `/chat/import` | required | Body `{ conversations: [...] }`. Idempotent guest→cloud import. |
| `GET` | `/chat/conversations` | required | User's cloud chat history. |
| `PUT` | `/chat/conversations/{id}/closed` | required | Body `{ is_closed: boolean }`; saves the tab visibility state. |
| `POST` | `/chat` | optional | Body: `text`, `chat_id`, `title`, message ids/timestamps, optional `history`/`context`/`effort`. `context` is a non-visible JSON object attached to this user message. Streams the legacy assistant response; persists when authenticated. |
| `POST` | `/chat/v2` | optional | Same body as `/chat`, with a 10,000-character message limit. `agent_v2` routes paper finding internally; other turns may fall back to legacy. |

## `library.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/library/papers` | required | All `user_papers` (incl. `origin`). `?saved=true` → only `is_saved`. |
| `PUT` | `/library/papers/{paper_id}/saved` | required | JSON `Paper` metadata. Sets `is_saved=true`. |
| `DELETE` | `/library/papers/{paper_id}/saved` | required | Unsave; deletes row only if no file. |
| `POST` | `/library/papers/saved` | required | Body `{ papers: Paper[] }` (max 100). Bulk upsert as saved. |
| `POST` | `/library/papers/import` | required | Body `{ items: [{ paper, raw? }] }` (max 100). `paper.title` and `paper.abstract` are required. Each valid item is saved as an independent `origin=user` paper with a `user:` ID; response has per-item `imported` / `invalid` results. |
| `PUT` | `/library/papers/{paper_id}/file` | required | multipart: `file` (PDF) + `metadata` (JSON `Paper`). |
| `DELETE` | `/library/papers/{paper_id}/file` | required | Deletes Azure file, clears upload fields; drops unsaved empty rows. |

## `notes.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/notes` | required | One research-notes document per user. Missing row → `200` with empty `content`. |
| `PUT` | `/notes` | required | Body `{ content: string }`. Upsert by `user_id`. |
