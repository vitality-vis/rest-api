import json
import config
from service import chroma
from logger_config import get_logger

# Use centralized logger
logging = get_logger()


class CachedData:

    umap_points = None
    meta_datas = None
    aggregated_metadata = None

    def init(self):
        logging.info("🔄 Initializing cached data...")

        # Cache all papers first (needed by other caching operations)
        chroma.load_all_papers_to_cache()

        # Cache UMAP points
        self.umap_points = chroma.get_all_umap_points()
        logging.info(f"✅ Loaded {len(self.umap_points) if self.umap_points else 0} UMAP points")

        # Load file-based metadata if available
        try:
            self.meta_datas = self.load_meta_data()
            logging.info("✅ Loaded meta_datas from file")
        except Exception as e:
            logging.warning(f"Could not load meta_data file: {e}")
            self.meta_datas = None

        # Cache aggregated metadata from ChromaDB
        logging.info("🔄 Computing aggregated metadata (this may take a moment for 75k papers)...")
        self.aggregated_metadata = self.compute_aggregated_metadata()
        logging.info("✅ Aggregated metadata cached successfully")

    def load_meta_data(self):
        with open(config.meta_data_file_path, 'r') as f:
            return json.load(f)

    def compute_aggregated_metadata(self):
        """Compute and cache all metadata aggregations at startup using cached papers (single fetch)"""
        # Use cached papers instead of fetching from ChromaDB 5 times
        docs = chroma.get_cached_papers()
        
        if not docs:
            logging.warning("No cached papers available, falling back to individual queries")
            return {
                'authors_summary': chroma.get_distinct_authors_with_counts(),
                'sources_summary': chroma.get_distinct_sources_with_counts(),
                'keywords_summary': chroma.get_distinct_keywords_with_counts(),
                'years_summary': chroma.get_distinct_years_with_counts(),
                'citation_counts': chroma.get_distinct_citation_counts()
            }
        
        logging.info(f"Computing aggregated metadata from {len(docs)} cached papers...")
        
        # Compute all aggregations from the single cached dataset
        def aggregate_count(field):
            counter = {}
            for doc in docs:
                values = doc.get(field)
                if values is None:
                    continue
                if not isinstance(values, list):
                    # For Authors and Keywords, split comma-separated strings into individual items
                    if field in ("Authors", "Keywords") and isinstance(values, str):
                        values = [v.strip() for v in values.split(",") if v.strip()]
                    else:
                        values = [values]
                for v in values:
                    if v:
                        key_str = str(v).strip()
                        if key_str:  # Only count non-empty strings
                            counter[key_str] = counter.get(key_str, 0) + 1
            return sorted([{"_id": k, "count": v} for k, v in counter.items()], key=lambda x: -x["count"])
        
        authors_summary = aggregate_count("Authors")
        sources_summary = aggregate_count("Source")
        keywords_summary = aggregate_count("Keywords")
        years_summary = sorted(aggregate_count("Year"), key=lambda x: x['_id'])
        citation_counts = sorted(set(doc.get("CitationCounts") for doc in docs if doc.get("CitationCounts") is not None))
        
        return {
            'authors_summary': authors_summary,
            'sources_summary': sources_summary,
            'keywords_summary': keywords_summary,
            'years_summary': years_summary,
            'citation_counts': citation_counts
        }

    def get_umap_points(self):
        return self.umap_points

    def get_meta_datas(self):
        return self.meta_datas

    def get_aggregated_metadata(self):
        """Return cached aggregated metadata"""
        return self.aggregated_metadata


cached_data = CachedData()