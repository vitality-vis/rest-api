import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import config
from logger_config import get_logger
from service import zilliz

logging = get_logger()


def main():
    logging.info("Exporting aggregated metadata from Zilliz...")
    metadata = zilliz.get_aggregated_metadata(sample_limit=None)

    logging.info("Exporting UMAP points from Zilliz...")
    umap_points = zilliz.get_all_umap_points()

    meta_path = Path(config.meta_data_file_path)
    umap_path = Path(config.umap_data_file_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    umap_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    umap_path.write_text(json.dumps(umap_points, ensure_ascii=False), encoding="utf-8")

    logging.info(
        "Export complete. metadata=%s umap_points=%s",
        meta_path,
        len(umap_points),
    )


if __name__ == "__main__":
    main()
