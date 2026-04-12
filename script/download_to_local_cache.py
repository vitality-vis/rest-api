#!/usr/bin/env python3
"""
Download data from Zilliz to local cache for faster startup.

Usage:
    python script/download_to_local_cache.py

This script will:
1. Download all papers from Zilliz
2. Download UMAP points from Zilliz
3. Download aggregated metadata from Zilliz
4. Save everything to data/local_cache/ directory

After running this script, you can start the server with:
    python main.py --use-local-cache
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import config
from logger_config import get_logger
from service import zilliz
from model.const import EMBED

logging = get_logger()

# ── Progress bar ──────────────────────────────────────────────

BAR_WIDTH = 40

def _print_progress(prefix, fetched, total):
    """Print an in-place terminal progress bar."""
    if total and total > 0:
        pct = fetched / total
        filled = int(BAR_WIDTH * pct)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        sys.stdout.write(f"\r  {prefix} |{bar}| {fetched}/{total} ({pct:.0%})")
    else:
        sys.stdout.write(f"\r  {prefix} ... {fetched} fetched")
    sys.stdout.flush()


def make_progress_callback(label):
    """Create a progress callback for a given download step."""
    def callback(phase, fetched, total):
        if phase == "ids":
            _print_progress(f"{label} [collecting IDs]", fetched, total)
        else:
            _print_progress(f"{label} [downloading]   ", fetched, total)
    return callback


# ── Helpers ───────────────────────────────────────────────────

def format_size(bytes_size):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def save_json(data, file_path, description):
    """Save data to JSON file with logging."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    file_path.write_text(json_str, encoding="utf-8")

    file_size = file_path.stat().st_size
    logging.info(f"  Saved {description} -> {file_path.name} ({format_size(file_size)})")
    return file_size


# ── Main ──────────────────────────────────────────────────────

def main():
    """Download all data from Zilliz to local cache."""
    start_time = datetime.now()
    print("=" * 60)
    print("  Downloading from Zilliz to local cache")
    print("=" * 60)

    total_size = 0
    embedding_type = EMBED.SPECTER

    # Create local_cache directory
    cache_dir = Path(config.local_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Cache dir: {cache_dir}\n")

    # ── Step 1: All papers ────────────────────────────────────
    print("📄 Step 1/3: Downloading all papers...")
    step_start = time.time()

    zilliz._all_papers_cache = {}  # Force fresh download
    zilliz.load_all_papers_to_cache(
        embedding_type,
        progress_callback=make_progress_callback("Papers"),
    )
    all_papers = zilliz.get_cached_papers(embedding_type)
    print()  # newline after progress bar

    elapsed = time.time() - step_start
    logging.info(f"  Downloaded {len(all_papers)} papers in {elapsed:.1f}s")
    total_size += save_json(
        all_papers,
        config.all_papers_cache_path,
        f"{len(all_papers)} papers",
    )
    print()

    # ── Step 2: UMAP points ──────────────────────────────────
    print("🗺️  Step 2/3: Downloading UMAP points...")
    step_start = time.time()

    umap_points = zilliz.get_all_umap_points(
        embedding_type,
        progress_callback=make_progress_callback("UMAP  "),
    )
    print()  # newline after progress bar

    elapsed = time.time() - step_start
    logging.info(f"  Downloaded {len(umap_points)} UMAP points in {elapsed:.1f}s")
    total_size += save_json(
        umap_points,
        config.umap_points_cache_path,
        f"{len(umap_points)} UMAP points",
    )
    print()

    # ── Step 3: Aggregated metadata ──────────────────────────
    print("📊 Step 3/3: Computing aggregated metadata...")

    authors_counter = Counter()
    sources_counter = Counter()
    keywords_counter = Counter()
    years_counter = Counter()
    titles_set = set()
    citation_counts_set = set()

    n = len(all_papers)
    for i, paper in enumerate(all_papers):
        authors = paper.get('Authors', [])
        if isinstance(authors, list):
            for a in authors:
                if a:
                    authors_counter[str(a).strip()] += 1
        source = paper.get('Source')
        if source:
            sources_counter[str(source).strip()] += 1
        keywords = paper.get('Keywords', [])
        if isinstance(keywords, list):
            for k in keywords:
                if k:
                    keywords_counter[str(k).strip()] += 1
        year = paper.get('Year')
        if year:
            years_counter[year] += 1
        title = paper.get('Title')
        if title:
            titles_set.add(str(title).strip())
        cc = paper.get('CitationCounts')
        if cc is not None:
            citation_counts_set.add(cc)

        if (i + 1) % 5000 == 0 or i + 1 == n:
            _print_progress("Metadata", i + 1, n)

    print()  # newline after progress bar

    metadata = {
        "authors_summary": sorted(
            [{"_id": k, "count": v} for k, v in authors_counter.items()],
            key=lambda x: -x["count"],
        ),
        "sources_summary": sorted(
            [{"_id": k, "count": v} for k, v in sources_counter.items()],
            key=lambda x: -x["count"],
        ),
        "keywords_summary": sorted(
            [{"_id": k, "count": v} for k, v in keywords_counter.items()],
            key=lambda x: -x["count"],
        ),
        "years_summary": sorted(
            [{"_id": k, "count": v} for k, v in years_counter.items()],
            key=lambda x: x["_id"],
        ),
        "titles": sorted(list(titles_set)),
        "citation_counts": sorted(list(citation_counts_set)),
    }

    logging.info(
        f"  Computed: {len(metadata['authors_summary'])} authors, "
        f"{len(metadata['sources_summary'])} sources, "
        f"{len(metadata['keywords_summary'])} keywords"
    )
    total_size += save_json(metadata, config.aggregated_metadata_cache_path, "aggregated metadata")

    # ── Summary ──────────────────────────────────────────────
    duration = (datetime.now() - start_time).total_seconds()
    print()
    print("=" * 60)
    print(f"  🎉  Download completed!")
    print(f"  Total size : {format_size(total_size)}")
    print(f"  Duration   : {duration:.1f}s")
    print(f"  Cache dir  : {cache_dir}")
    print()
    print("  Start the server with:")
    print("    python main.py --use-local-cache")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n\n❌ Download failed: {e}", exc_info=True)
        sys.exit(1)
