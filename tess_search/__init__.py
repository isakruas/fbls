"""CUDA float64 transit-search interfaces."""
__version__ = "0.4.0"

from .api import SearchConfig, TransitSearchResult, search_prepared, search_transits
from .core import (NoiseModel, PreparedCurve, SearchResult, period_grid, prepare_families, detrending_windows,
                   search as search_templates)

__all__ = ["SearchConfig", "TransitSearchResult", "search_transits", "search_prepared",
           "NoiseModel", "PreparedCurve", "SearchResult", "prepare_families", "detrending_windows",
           "period_grid", "search_templates"]

