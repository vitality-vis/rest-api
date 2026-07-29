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
| `POST` | `/chat` | optional | Body: `text`, `chat_id`, `title`, message ids/timestamps, optional `history`/`effort`. Streams the legacy assistant response; persists when authenticated. |
| `POST` | `/chat/v2` | optional | Same body as `/chat`, with a 10,000-character message limit. Experimental route: explicit paper-finding turns use search v2; all other turns fall back to legacy. |

## `search_v2.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `POST` | `/search/v2` | optional | Body: `query` (1–10,000 characters), optional `effort` (defaults to `low`) and `result_limit`. Returns synchronous low-effort search results; a request without a topic or metadata filter returns 400. |

## `library.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/library/papers` | required | All `user_papers`. `?saved=true` → only `is_saved`. |
| `POST` | `/library/papers/import` | required | Body `{ papers: Paper[] }` (max 100). Upsert as saved. |
| `PUT` | `/library/papers/{paper_id}/saved` | required | JSON `Paper` metadata. Sets `is_saved=true`. |
| `DELETE` | `/library/papers/{paper_id}/saved` | required | Unsave; deletes row only if no file. |
| `PUT` | `/library/papers/{paper_id}/file` | required | multipart: `file` (PDF) + `metadata` (JSON `Paper`). |
| `DELETE` | `/library/papers/{paper_id}/file` | required | Deletes Azure file, clears upload fields; drops unsaved empty rows. |

## `notes.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/notes` | required | One research-notes document per user. Missing row → `200` with empty `content`. |
| `PUT` | `/notes` | required | Body `{ content: string }`. Upsert by `user_id`. |
