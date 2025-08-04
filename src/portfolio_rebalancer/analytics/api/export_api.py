"""Export and integration API endpoints for analytics results."""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, status
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field, validator
import uuid

from ..analytics_service import AnalyticsService
from ..export import (
    AnalyticsExporter, WebhookNotifier, BulkAnalyticsProcessor,
    ExportConfig, ExportFormat, CompressionType, WebhookConfig,
    PaginationConfig, StreamingConfig
)
from ..models import (
    BacktestResult, MonteCarloResult, RiskAnalysis, 
    PerformanceMetrics, DividendAnalysis
)
from ..exceptions import ExportError, AnalyticsError
from .auth import get_current_active_user, TokenData, require_permission, Permission

logger = logging.getLogger(__name__)

# API router
router = APIRouter(prefix="/export", tags=["export"])


# Request/Response Models
class ExportRequest(BaseModel):
    """Request model for data export."""
    
    result_id: str = Field(..., description="Result identifier to export")
    result_type: str = Field(..., description="Type of result (backtest, monte_carlo, etc.)")
    format: ExportFormat = Field(ExportFormat.JSON, description="Export format")
    include_raw_data: bool = Field(False, description="Include raw data in export")
    include_statistics: bool = Field(True, description="Include statistics")
    include_percentiles: bool = Field(True, description="Include percentiles (Monte Carlo)")
    include_metadata: bool = Field(True, description="Include metadata")
    compression: Optional[CompressionType] = Field(None, description="Compression type")
    decimal_places: int = Field(4, ge=0, le=10, description="Decimal places for numbers")
    
    @validator('result_type')
    def validate_result_type(cls, v):
        valid_types = ['backtest', 'monte_carlo', 'risk_analysis', 'performance', 'dividend_analysis']
        if v not in valid_types:
            raise ValueError(f"Result type must be one of {valid_types}")
        return v


class BulkExportRequest(BaseModel):
    """Request model for bulk data export."""
    
    result_ids: List[str] = Field(..., min_items=1, description="List of result identifier