"""CUDA float64 transit-search interfaces."""
__version__ = "0.1.0"

from .api import SearchConfig, TransitSearchResult, search_prepared, search_transits
from .core import NoiseModel, PreparedCurve, SearchResult, period_grid, prepare_curve, search as search_templates

__all__ = ["SearchConfig", "TransitSearchResult", "search_transits", "search_prepared",
           "NoiseModel", "PreparedCurve", "SearchResult", "prepare_curve", "period_grid", "search_templates"]

