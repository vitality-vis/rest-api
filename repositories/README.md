# Repositories

A repository is the boundary between application use cases and persistent
data. Its public functions describe the data the application needs, while the
database client, collection names, scalar expressions, batching, and result
normalisation remain implementation details.

Current layout:

```text
repositories/
  zilliz/
    paper_repository.py    # paper lookup, filtering, and vector retrieval
    connection.py          # pymilvus connection and collection lifecycle
    query_expressions.py   # Milvus filter-expression compilation
    mappers.py             # Zilliz row <-> internal paper mapping
```

`zilliz/paper_repository.py` is intentionally a compatibility facade during
the migration. New HTTP routes and services should import it instead of
`service.zilliz`. Once all callers have moved, the legacy implementation can
be split into the modules listed above and removed without changing callers.
