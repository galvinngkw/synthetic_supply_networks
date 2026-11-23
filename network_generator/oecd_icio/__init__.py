"""Reproducible pipeline from OECD ICIO source bundles to the per-country tables in ``data/``.

See ``build.py`` for the recipe and ``__main__.py`` for the command line.
"""

from network_generator.oecd_icio.build import (
    Release,
    build,
    compare,
    load_releases,
    total_use_by_country,
)

__all__ = ["Release", "build", "compare", "load_releases", "total_use_by_country"]
