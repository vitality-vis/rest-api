"""Tests for service.static_cache.

Unit tests mock Zilliz and never touch real data/*.json.
live tests are read-only (fingerprint only).

  pytest tests/test_static_cache.py -q
  pytest tests/test_static_cache.py -m live -q
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from service.static_cache import (
    CachedData,
    fingerprints_match,
    get_aggregated_metadata,
    read_collection_fingerprint,
    write_static_cache_from_zilliz,
    ZillizFingerprintError,
    ZillizNotConfiguredError,
)


def _empty_meta():
    return {
        "authors_summary": [],
        "sources_summary": [],
        "keywords_summary": [],
        "years_summary": [],
        "citation_counts": [],
    }


def _patch_cache_paths(monkeypatch, meta_path, umap_path, fingerprint_path):
    monkeypatch.setattr("service.static_cache.config.meta_data_file_path", str(meta_path))
    monkeypatch.setattr("service.static_cache.config.umap_data_file_path", str(umap_path))
    monkeypatch.setattr(
        "service.static_cache.config.cache_fingerprint_file_path",
        str(fingerprint_path),
    )


def test_fingerprints_match_happy_and_mismatch():
    base = {
        "collection": "paper_specter",
        "update_timestamp": 100,
        "row_count": 10,
    }
    assert fingerprints_match(base, dict(base)) is True
    assert fingerprints_match(base, {**base, "row_count": 11}) is False
    assert fingerprints_match(base, {**base, "update_timestamp": 99}) is False
    assert fingerprints_match(base, {**base, "collection": "other"}) is False
    assert fingerprints_match(None, base) is False
    assert fingerprints_match(base, None) is False


def test_read_collection_fingerprint_uses_fallback_fields(monkeypatch):
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def describe_collection(self, **_kwargs):
            return {"updated_timestamp": "42"}

        def get_collection_stats(self, **_kwargs):
            return {"rowCount": "3"}

    monkeypatch.setattr("service.static_cache.config.ZILLIZ_URI", "https://example")
    monkeypatch.setattr("service.static_cache.config.ZILLIZ_TOKEN", "token")
    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        SimpleNamespace(MilvusClient=FakeClient),
    )

    assert read_collection_fingerprint() == {
        "collection": "paper_specter",
        "update_timestamp": 42,
        "row_count": 3,
    }


def test_read_collection_fingerprint_raises_when_no_change_detector(monkeypatch):
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def describe_collection(self, **_kwargs):
            return {}

        def get_collection_stats(self, **_kwargs):
            return {}

    monkeypatch.setattr("service.static_cache.config.ZILLIZ_URI", "https://example")
    monkeypatch.setattr("service.static_cache.config.ZILLIZ_TOKEN", "token")
    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        SimpleNamespace(MilvusClient=FakeClient),
    )

    with pytest.raises(ZillizFingerprintError, match="neither update_timestamp nor row_count"):
        read_collection_fingerprint()


def test_get_aggregated_metadata_counts_authors(monkeypatch):
    docs = [
        {
            "Title": "T1",
            "Authors": ["A", "B"],
            "Keywords": [],
            "Source": "CHI",
            "Year": 2020,
            "CitationCounts": 1,
        },
        {
            "Title": "T2",
            "Authors": ["A"],
            "Keywords": ["viz"],
            "Source": "CHI",
            "Year": 2021,
            "CitationCounts": 2,
        },
    ]
    monkeypatch.setattr(
        "service.static_cache.zilliz.get_all_metadatas",
        get_all_metadatas := Mock(return_value=docs),
    )
    meta = get_aggregated_metadata(sample_limit=None)
    authors = {row["_id"]: row["count"] for row in meta["authors_summary"]}
    assert authors["A"] == 2
    assert authors["B"] == 1
    assert meta["sources_summary"] == [{"_id": "CHI", "count": 2}]
    assert meta["keywords_summary"] == [{"_id": "viz", "count": 1}]
    assert meta["years_summary"] == [
        {"_id": "2020", "count": 1},
        {"_id": "2021", "count": 1},
    ]
    assert meta["citation_counts"] == [1, 2]
    assert meta["sample_size"] == 2
    get_all_metadatas.assert_called_once_with("specter", limit=None)


def test_write_static_cache_from_zilliz_writes_normalized_snapshot(tmp_path, monkeypatch):
    meta_p = tmp_path / "meta_data.json"
    umap_p = tmp_path / "umap_data.json"
    fp_p = tmp_path / "cache_fingerprint.json"
    fingerprint = {
        "collection": "paper_specter",
        "update_timestamp": 42,
        "row_count": 1,
    }
    rows = [
        {
            "ID": "1",
            "Title": "T1",
            "Authors": ["A", "B"],
            "Keywords": ["viz"],
            "Source": "CHI",
            "Year": 2024,
            "CitationCounts": 1,
        }
    ]
    points = [{"ID": "1", "x": 0.1, "y": 0.2}]
    _patch_cache_paths(monkeypatch, meta_p, umap_p, fp_p)
    get_cache_rows = Mock(return_value=rows)
    format_umap_points = Mock(return_value=points)
    monkeypatch.setattr(
        "service.static_cache.zilliz.get_all_static_cache_rows", get_cache_rows
    )
    monkeypatch.setattr(
        "service.static_cache.zilliz.format_umap_points", format_umap_points
    )

    result = write_static_cache_from_zilliz(fingerprint=fingerprint)

    expected_metadata = {
        **_empty_meta(),
        "authors_summary": [{"_id": "A", "count": 1}, {"_id": "B", "count": 1}],
        "keywords_summary": [{"_id": "viz", "count": 1}],
        "sources_summary": [{"_id": "CHI", "count": 1}],
        "years_summary": [{"_id": "2024", "count": 1}],
        "citation_counts": [1],
        "sample_size": 1,
    }
    assert result == {
        "metadata": expected_metadata,
        "umap_points": points,
        "fingerprint": fingerprint,
    }
    assert json.loads(meta_p.read_text(encoding="utf-8")) == expected_metadata
    assert json.loads(umap_p.read_text(encoding="utf-8")) == points
    assert json.loads(fp_p.read_text(encoding="utf-8")) == fingerprint
    get_cache_rows.assert_called_once_with("specter")
    format_umap_points.assert_called_once_with(rows)


def test_write_static_cache_rejects_incomplete_zilliz_results(tmp_path, monkeypatch):
    meta_p = tmp_path / "meta_data.json"
    umap_p = tmp_path / "umap_data.json"
    fp_p = tmp_path / "cache_fingerprint.json"
    _patch_cache_paths(monkeypatch, meta_p, umap_p, fp_p)
    meta_p.write_text('{"existing": true}', encoding="utf-8")
    umap_p.write_text('[{"ID": "existing"}]', encoding="utf-8")
    fp_p.write_text('{"collection": "previous"}', encoding="utf-8")

    monkeypatch.setattr(
        "service.static_cache.zilliz.get_all_static_cache_rows",
        Mock(return_value=[{"ID": "1"}]),
    )

    with pytest.raises(RuntimeError, match="expected 2 rows, received 1"):
        write_static_cache_from_zilliz(
            fingerprint={
                "collection": "paper_specter",
                "update_timestamp": 42,
                "row_count": 2,
            }
        )

    assert meta_p.read_text(encoding="utf-8") == '{"existing": true}'
    assert umap_p.read_text(encoding="utf-8") == '[{"ID": "existing"}]'
    assert fp_p.read_text(encoding="utf-8") == '{"collection": "previous"}'


def test_cached_data_init_uses_local_when_fingerprint_matches(tmp_path, monkeypatch):
    meta_p = tmp_path / "meta_data.json"
    umap_p = tmp_path / "umap_data.json"
    fp_p = tmp_path / "cache_fingerprint.json"
    fp = {"collection": "paper_specter", "update_timestamp": 42, "row_count": 3}

    meta_p.write_text(json.dumps(_empty_meta()), encoding="utf-8")
    umap_p.write_text(json.dumps([{"ID": "1", "Title": "x"}]), encoding="utf-8")
    fp_p.write_text(json.dumps(fp), encoding="utf-8")

    _patch_cache_paths(monkeypatch, meta_p, umap_p, fp_p)
    monkeypatch.setattr(
        "service.static_cache.get_collection_fingerprint",
        lambda *a, **k: fp,
    )

    with patch("service.static_cache.write_static_cache_from_zilliz") as refresh:
        cache = CachedData()
        cache.init()
        refresh.assert_not_called()
        assert len(cache.get_umap_points()) == 1
        assert cache.fingerprint == fp


def test_cached_data_init_refreshes_when_stale(tmp_path, monkeypatch):
    meta_p = tmp_path / "meta_data.json"
    umap_p = tmp_path / "umap_data.json"
    fp_p = tmp_path / "cache_fingerprint.json"
    local = {"collection": "paper_specter", "update_timestamp": 1, "row_count": 1}
    remote = {"collection": "paper_specter", "update_timestamp": 2, "row_count": 99}

    meta_p.write_text(json.dumps(_empty_meta()), encoding="utf-8")
    umap_p.write_text(json.dumps([]), encoding="utf-8")
    fp_p.write_text(json.dumps(local), encoding="utf-8")

    _patch_cache_paths(monkeypatch, meta_p, umap_p, fp_p)
    monkeypatch.setattr(
        "service.static_cache.get_collection_fingerprint",
        lambda *a, **k: remote,
    )

    def fake_write(embedding_type, fingerprint=None):
        meta_p.write_text(
            json.dumps(_empty_meta()),
            encoding="utf-8",
        )
        umap_p.write_text(json.dumps([{"ID": "9"}]), encoding="utf-8")
        fp_p.write_text(json.dumps(fingerprint or remote), encoding="utf-8")
        return {
            "metadata": {},
            "umap_points": [{"ID": "9"}],
            "fingerprint": fingerprint or remote,
        }

    monkeypatch.setattr(
        "service.static_cache.write_static_cache_from_zilliz", fake_write
    )

    cache = CachedData()
    cache.init()
    assert cache.get_umap_points()[0]["ID"] == "9"
    assert cache.fingerprint == remote


def test_cached_data_init_uses_local_cache_when_remote_fingerprint_unavailable(
    tmp_path, monkeypatch
):
    meta_p = tmp_path / "meta_data.json"
    umap_p = tmp_path / "umap_data.json"
    fp_p = tmp_path / "cache_fingerprint.json"
    local_fp = {"collection": "paper_specter", "update_timestamp": 1, "row_count": 1}
    points = [{"ID": "local"}]
    meta_p.write_text(json.dumps(_empty_meta()), encoding="utf-8")
    umap_p.write_text(json.dumps(points), encoding="utf-8")
    fp_p.write_text(json.dumps(local_fp), encoding="utf-8")
    _patch_cache_paths(monkeypatch, meta_p, umap_p, fp_p)
    monkeypatch.setattr(
        "service.static_cache.get_collection_fingerprint", lambda *_args, **_kwargs: None
    )
    refresh = Mock()
    monkeypatch.setattr("service.static_cache.write_static_cache_from_zilliz", refresh)

    cache = CachedData()
    cache.init()

    assert cache.get_umap_points() == points
    assert cache.fingerprint == local_fp
    refresh.assert_not_called()


def test_cached_data_init_keeps_local_cache_when_refresh_fails(tmp_path, monkeypatch):
    meta_p = tmp_path / "meta_data.json"
    umap_p = tmp_path / "umap_data.json"
    fp_p = tmp_path / "cache_fingerprint.json"
    local_fp = {"collection": "paper_specter", "update_timestamp": 1, "row_count": 1}
    remote_fp = {"collection": "paper_specter", "update_timestamp": 2, "row_count": 2}
    points = [{"ID": "local"}]
    meta_p.write_text(json.dumps(_empty_meta()), encoding="utf-8")
    umap_p.write_text(json.dumps(points), encoding="utf-8")
    fp_p.write_text(json.dumps(local_fp), encoding="utf-8")
    _patch_cache_paths(monkeypatch, meta_p, umap_p, fp_p)
    monkeypatch.setattr(
        "service.static_cache.get_collection_fingerprint",
        lambda *_args, **_kwargs: remote_fp,
    )
    refresh = Mock(side_effect=RuntimeError("Zilliz unavailable"))
    monkeypatch.setattr("service.static_cache.write_static_cache_from_zilliz", refresh)

    cache = CachedData()
    cache.init()

    assert cache.get_umap_points() == points
    assert cache.fingerprint == local_fp
    refresh.assert_called_once_with("specter", fingerprint=remote_fp)


def test_cached_data_init_is_empty_when_no_local_cache_or_remote_fingerprint(
    tmp_path, monkeypatch
):
    _patch_cache_paths(
        monkeypatch,
        tmp_path / "meta_data.json",
        tmp_path / "umap_data.json",
        tmp_path / "cache_fingerprint.json",
    )
    monkeypatch.setattr(
        "service.static_cache.get_collection_fingerprint", lambda *_args, **_kwargs: None
    )

    cache = CachedData()
    cache.init()

    assert cache.get_umap_points() == []
    assert cache.get_meta_datas() == {}
    assert cache.get_aggregated_metadata() == {}
    assert cache.fingerprint is None


@pytest.mark.live
def test_zilliz_fingerprint_smoke_readonly():
    """Live read-only: describe/stats only. Does not write data/*.json."""
    try:
        fp = read_collection_fingerprint()
    except ZillizNotConfiguredError:
        pytest.skip("ZILLIZ_URI / ZILLIZ_TOKEN not set")

    print(f"Zilliz fingerprint: {json.dumps(fp, ensure_ascii=False)}")
    assert fp.get("collection") == "paper_specter"
    assert fp.get("update_timestamp") is not None or fp.get("row_count") is not None
