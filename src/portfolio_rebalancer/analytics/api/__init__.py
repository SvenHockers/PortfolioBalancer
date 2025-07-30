"""Analytics API module."""

from .performance_api import router as performance_router
from .dividend_api import router as dividend_router

__all__ = ['performance_router', 'dividend_router']