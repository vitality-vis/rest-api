import json
import config
from logger_config import get_logger
from service.metadata_normalizer import normalize_aggregated_metadata

# Use centralized logger
logging = get_logger()


class CachedData:

    umap_points = None
    meta_datas = None
    aggregated_metadata = None

    def init(self):
        logging.info("Initializing cached data...")

        self.umap_points = self.load_json_file(config.umap_data_file_path, [])
        logging.info(f"Loaded {len(self.umap_points) if self.umap_points else 0} UMAP points from file")

        self.meta_datas = self.load_json_file(config.meta_data_file_path, {})
        self.aggregated_metadata = normalize_aggregated_metadata(self.meta_datas)
        logging.info("Loaded aggregated metadata from file")

    def load_json_file(self, path, default):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load JSON file {path}: {e}")
            return default

    def get_umap_points(self):
        return self.umap_points

    def get_meta_datas(self):
        return self.meta_datas

    def get_aggregated_metadata(self):
        """Return cached aggregated metadata"""
        return self.aggregated_metadata


cached_data = CachedData()
