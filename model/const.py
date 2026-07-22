class EMBED:
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    DEFAULT = TEXT_EMBEDDING_3_SMALL

    # Retained only so callers can produce a clear unsupported-model response.
    # These are not registered retrieval profiles.
    SPECTER = "specter"
    ADA = "ada"

    ALL = {TEXT_EMBEDDING_3_SMALL}
