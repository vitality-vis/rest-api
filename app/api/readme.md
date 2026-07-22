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
| `POST` | `/chat` | optional | Body: `text`, `chat_id`, `title`, message ids/timestamps, optional `history`. Streams assistant response; persists when authenticated. |

## `library.py`

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `GET` | `/library/papers` | required | All `user_papers`. `?saved=true` → only `is_saved`. |
| `POST` | `/library/papers/import` | required | Body `{ papers: Paper[] }` (max 100). Upsert as saved. |
| `PUT` | `/library/papers/{paper_id}/saved` | required | JSON `Paper` metadata. Sets `is_saved=true`. |
| `DELETE` | `/library/papers/{paper_id}/saved` | required | Unsave; deletes row only if no file. |
| `PUT` | `/library/papers/{paper_id}/file` | required | multipart: `file` (PDF) + `metadata` (JSON `Paper`). |
| `DELETE` | `/library/papers/{paper_id}/file` | required | Deletes Azure file, clears upload fields; drops unsaved empty rows. |
