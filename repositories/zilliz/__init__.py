"""Zilliz-backed repository implementations.

Import a concrete module (for example ``zilliz.paper_repository``) rather
than re-exporting it here.  This keeps the connection module usable by legacy
code during the staged migration without creating an import cycle.
"""
