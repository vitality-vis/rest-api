import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

config.load_project_environment()

# Must run before importing service.* (those call get_logger on import).
from service.bootstrap import initialize_runtime

logging = initialize_runtime(enable_gcp=False)

from model.const import EMBED
from service.static_cache import write_static_cache_from_zilliz


def main():
    logging.info("Starting static cache export from Zilliz...")
    result = write_static_cache_from_zilliz(embedding_type=EMBED.SPECTER)
    logging.info(
        "Export complete. umap_points=%s sample_size=%s fingerprint=%s",
        len(result["umap_points"]),
        (result["metadata"] or {}).get("sample_size"),
        result["fingerprint"],
    )


if __name__ == "__main__":
    main()
