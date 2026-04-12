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
    all_papers = None  # New: cache for all papers

    def init(self, use_local_cache=False):
        """
        Initialize cached data from files.

        Args:
            use_local_cache: If True, load from local_cache directory instead of Zilliz
        """
        logging.info("Initializing cached data...")

        if use_local_cache:
            logging.info("🚀 Using local cache files (faster startup)")
            # Load from local_cache directory
            self.umap_points = self.load_json_file(config.umap_points_cache_path, [])
            logging.info(f"Loaded {len(self.umap_points) if self.umap_points else 0} UMAP points from local cache")

            self.meta_datas = self.load_json_file(config.aggregated_metadata_cache_path, {})
            self.aggregated_metadata = self.meta_datas  # Already normalized in cache
            logging.info("Loaded aggregated metadata from local cache")

            self.all_papers = self.load_json_file(config.all_papers_cache_path, [])
            logging.info(f"Loaded {len(self.all_papers) if self.all_papers else 0} papers from local cache")
        else:
            logging.info("📡 Loading from original data files (will connect to Zilliz for papers)")
            # Original behavior: load from data directory
            self.umap_points = self.load_json_file(config.umap_data_file_path, [])
            logging.info(f"Loaded {len(self.umap_points) if self.umap_points else 0} UMAP points from file")

            self.meta_datas = self.load_json_file(config.meta_data_file_path, {})
            self.aggregated_metadata = normalize_aggregated_metadata(self.meta_datas)
            logging.info("Loaded aggregated metadata from file")

            # Don't load all_papers here - will be loaded from Zilliz on demand
            self.all_papers = None

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

    def get_all_papers(self):
        """Return cached all papers (only available if use_local_cache=True)"""
        return self.all_papers


cached_data = CachedData()
