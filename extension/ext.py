import json

import config
from service import marqo


class CachedData:

    umap_points = None
    meta_datas = None

    def init(self):
        self.umap_points = marqo.query_all_umap_points()
        self.meta_datas = self.load_meta_data()

    def load_meta_data(self):
        with open(config.meta_data_file_path, 'r') as f:
            return json.load(f)

    def get_umap_points(self):
        return self.umap_points

    def get_meta_datas(self):
        return self.meta_datas


cached_data = CachedData()
