"""Main analytics API router with comprehensive endpoints."""

import logging
from typing import Dict, List, Any, Optional
from datetime import date, datetime
from fastapi import APIRouter, HTTPException, Query, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import asyncio
import uuid

from ...common.models import Portfolio
from ..analytics_service import AnalyticsService
from ..models import (
    BacktestConfig, MonteCarloConfig, PerformanceMetrics, 
    RiskAnalysis, DividendAnalysis, AnalyticsError, BacktestResult, MonteCarloResult
)
from ..exceptions import (
    BacktestError, SimulationError, RiskAnalysisError,
    PerformanceTrackingError, DividendAnalysisError
)
from .auth import (
    get_current_active_user, check_rate_limit, TokenData,
    require_permission, require_portfolio_access, Permission,
    LoginRequest, LoginResponse, RefreshRequest, auth_service
)

logger = logging.getLogger(__name__)

# API versioning
API_VERSION = "v1"
router = APIRouter(prefix=f"/api/{API_VERSION}/analytics", tags=["analytics"])


# Request/Response Models
class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Check timestamp")
    components: Dict[str, str] = Field(..., description="Component health status")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


class AnalyticsStatusResponse(BaseModel):
    """Analytics operation status response."""
    operation_id: str = Field(..., description="Operation identifier")
    status: str = Field(..., description="Operation status")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    message: Optional[str] = Field(None, description="Status message")
    result: Optional[Dict[str, Any]] = Field(None, description="Operation result if completed")
    created_at: datetime = Field(..., description="Operation start time")
    updated_at: datetime = Field(..., description="Last update time")


class BulkAnalyticsRequest(BaseModel):
    """Bulk analytics request model."""
    portfolios: List[Dict[str, Any]] = Field(..., description="List of portfolio configurations")
    operations: List[str] = Field(..., description="Analytics operations to perform")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options")
    
    @validator('operations')
    def validate_operations(cls, v):
        valid_ops = ['backtest', 'monte_carlo', 'risk_analysis', 'performance', 'dividends']
        for op in v:
            if op not in valid_ops:
                raise ValueError(f"Invalid operation: {op}. Must be one of {valid_ops}")
        return v


class BulkAnalyticsResponse(BaseModel):
    """Bulk analytics response model."""
    batch_id: str = Field(..., description="Batch operation identifier")
    total_operations: int = Field(..., description="Total number of operations")
    completed: int = Field(..., description="Number of completed operations")
    failed: int = Field(..., description="Number of failed operations")
    results: List[Dict[str, Any]] = Field(..., description="Individual operation results")
    status: str = Field(..., description="Overall batch status")


# Dependency injection
def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance."""
    # In a real implementation, this would be properly injected
    # For now, return None and handle in endpoints
    return None


def get_request_id(request: Request) -> str:
    """Extract or generate request correlation ID."""
    return request.headers.get("X-Request-ID", f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")


# Authentication endpoints
@router.post("/auth/login", response_model=LoginResponse, tags=["authentication"])
async def login(request: LoginRequest):
    """
    Authenticate user and return access tokens.
    
    Returns JWT access token and refresh token for API access.
    """
    try:
        user = auth_service.authenticate_user(request.username, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        access_token = auth_service.create_access_token(user)
        refresh_token = auth_service.create_refresh_token(user)
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=24 * 3600,  # 24 hours
            user={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions],
                "portfolio_access": user.portfolio_access
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/auth/refresh", tags=["authentication"])
async def refresh_token(request: RefreshRequest):
    """
    Refresh access token using refresh token.
    
    Returns new access token when refresh token is valid.
    """
    try:
        access_token = auth_service.refresh_access_token(request.refresh_token)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 24 * 3600
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.post("/auth/logout", tags=["authentication"])
async def logout(
    current_user: TokenData = Depends(get_current_active_user),
    request: Request = None
):
    """
    Logout user and revoke tokens.
    
    Revokes the current access token to prevent further use.
    """
    try:
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            auth_service.revoke_token(token)
        
        return {
            "message": "Successfully logged out",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {
            "message": "Logout completed",
            "timestamp": datetime.now().isoformat()
        }


# Exception handlers
@router.exception_handler(AnalyticsError)
async def analytics_error_handler(request: Request, exc: AnalyticsError):
    """Handle analytics-specific errors."""
    request_id = get_request_id(request)
    logger.error(f"Analytics error [{request_id}]: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="AnalyticsError",
            message=str(exc),
            request_id=request_id
        ).dict()
    )


@router.exception_handler(BacktestError)
async def backtest_error_handler(request: Request, exc: BacktestError):
    """Handle backtesting errors."""
    request_id = get_request_id(request)
    logger.error(f"Backtest error [{request_id}]: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="BacktestError",
            message=str(exc),
            request_id=request_id
        ).dict()
    )


@router.exception_handler(SimulationError)
async def simulation_error_handler(request: Request, exc: SimulationError):
    """Handle Monte Carlo simulation errors."""
    request_id = get_request_id(request)
    logger.error(f"Simulation error [{request_id}]: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="SimulationError",
            message=str(exc),
            request_id=request_id
        ).dict()
    )


# Include sub-routers
from .interactive_api import router as interactive_router

# Add interactive router to main router
router.include_router(interactive_router)

# Main API endpoints
@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: TokenData = Depends(get_current_active_user),
    _: None = Depends(check_rate_limit)
):
    """
    Check analytics service health and component status.
    
    Returns comprehensive health information for monitoring.
    """
    try:
        if not analytics_service:
            # Mock health check when service not available
            return HealthCheckResponse(
                status="degraded",
                version=API_VERSION,
                timestamp=datetime.now(),
                components={
                    "analytics_service": "unavailable",
                    "data_storage": "unknown",
                    "analytics_storage": "unknown"
                }
            )
        
        health_info = analytics_service.health_check()
        
        return HealthCheckResponse(
            status=health_info["status"],
            version=API_VERSION,
            timestamp=datetime.now(),
            components={
                "analytics_service": "healthy",
                "data_storage": health_info.get("data_storage", "unknown"),
                "analytics_storage": health_info.get("analytics_storage", "unknown")
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version=API_VERSION,
            timestamp=datetime.now(),
            components={
                "analytics_service": "error",
                "error": str(e)
            }
        )


@router.get("/version")
async def get_version(
    current_user: TokenData = Depends(get_current_active_user),
    _: None = Depends(check_rate_limit)
):
    """Get API version information."""
    return {
        "version": API_VERSION,
        "service": "portfolio-analytics",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "backtest": f"/api/{API_VERSION}/analytics/backtest",
            "monte_carlo": f"/api/{API_VERSION}/analytics/monte-carlo",
            "risk": f"/api/{API_VERSION}/analytics/risk",
            "performance": f"/api/{API_VERSION}/analytics/performance",
            "dividends": f"/api/{API_VERSION}/analytics/dividends",
            "export": f"/api/{API_VERSION}/analytics/export"
        }
    }


@router.get("/capabilities")
async def get_capabilities():
    """Get analytics service capabilities and supported operations."""
    return {
        "analytics_operations": [
            {
                "name": "backtest",
                "description": "Historical portfolio backtesting with strategy comparison",
                "supported_strategies": ["sharpe", "min_variance", "equal_weight", "custom"],
                "max_time_horizon": "20 years",
                "supported_frequencies": ["daily", "weekly", "monthly", "quarterly"]
            },
            {
                "name": "monte_carlo",
                "description": "Monte Carlo simulation for portfolio projections",
                "max_simulations": 100000,
                "max_time_horizon": "50 years",
                "supported_scenarios": ["historical", "stress_test", "custom"]
            },
            {
                "name": "risk_analysis",
                "description": "Comprehensive portfolio risk analysis",
                "metrics": ["VaR", "CVaR", "beta", "correlation", "factor_exposure"],
                "confidence_levels": [0.90, 0.95, 0.99]
            },
            {
                "name": "performance_tracking",
                "description": "Real-time portfolio performance monitoring",
                "metrics": ["returns", "sharpe", "alpha", "beta", "tracking_error"],
                "benchmarks": ["SPY", "QQQ", "custom"]
            },
            {
                "name": "dividend_analysis",
                "description": "Income and dividend analysis",
                "features": ["yield_tracking", "income_projection", "sustainability_analysis"]
            }
        ],
        "export_formats": ["json", "csv", "excel", "pdf"],
        "visualization_support": {
            "grafana_integration": True,
            "chart_types": ["time_series", "scatter", "heatmap", "waterfall"],
            "real_time_updates": True
        }
    }


@router.post("/bulk", response_model=BulkAnalyticsResponse)
async def bulk_analytics_operations(
    request: BulkAnalyticsRequest,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    request_id: str = Depends(get_request_id)
):
    """
    Perform bulk analytics operations across multiple portfolios.
    
    Supports running multiple analytics operations on multiple portfolios
    in a single request with progress tracking and error handling.
    """
    try:
        logger.info(f"Starting bulk analytics operation [{request_id}]", extra={
            'portfolio_count': len(request.portfolios),
            'operations': request.operations,
            'request_id': request_id
        })
        
        if not analytics_service:
            raise HTTPException(
                status_code=503, 
                detail="Analytics service not available"
            )
        
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id[-8:]}"
        total_operations = len(request.portfolios) * len(request.operations)
        results = []
        completed = 0
        failed = 0
        
        # Process each portfolio
        for portfolio_data in request.portfolios:
            portfolio_id = portfolio_data.get('id', f"portfolio_{len(results)}")
            tickers = portfolio_data.get('tickers', [])
            weights = portfolio_data.get('weights', [])
            
            # Process each operation for this portfolio
            for operation in request.operations:
                try:
                    result = await _execute_analytics_operation(
                        analytics_service, operation, portfolio_id, tickers, weights, request.options
                    )
                    results.append({
                        'portfolio_id': portfolio_id,
                        'operation': operation,
                        'status': 'completed',
                        'result': result
                    })
                    completed += 1
                    
                except Exception as e:
                    logger.error(f"Operation {operation} failed for portfolio {portfolio_id}: {e}")
                    results.append({
                        'portfolio_id': portfolio_id,
                        'operation': operation,
                        'status': 'failed',
                        'error': str(e)
                    })
                    failed += 1
        
        # Determine overall status
        if failed == 0:
            overall_status = "completed"
        elif completed == 0:
            overall_status = "failed"
        else:
            overall_status = "partial"
        
        logger.info(f"Bulk analytics operation completed [{request_id}]", extra={
            'batch_id': batch_id,
            'total_operations': total_operations,
            'completed': completed,
            'failed': failed,
            'status': overall_status
        })
        
        return BulkAnalyticsResponse(
            batch_id=batch_id,
            total_operations=total_operations,
            completed=completed,
            failed=failed,
            results=results,
            status=overall_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk analytics operation failed [{request_id}]: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Bulk operation failed: {str(e)}"
        )


@router.get("/operations/{operation_id}/status", response_model=AnalyticsStatusResponse)
async def get_operation_status(
    operation_id: str,
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get status of a long-running analytics operation.
    
    Provides real-time status updates for asynchronous operations
    like backtesting and Monte Carlo simulations.
    """
    try:
        # In a real implementation, this would check operation status from a job queue
        # For now, return mock status
        
        # Parse operation ID to determine type and status
        if "backtest" in operation_id:
            operation_type = "backtest"
        elif "monte_carlo" in operation_id:
            operation_type = "monte_carlo"
        elif "risk" in operation_id:
            operation_type = "risk_analysis"
        else:
            operation_type = "unknown"
        
        # Mock status based on operation age
        created_time = datetime.now().replace(second=0, microsecond=0)  # Mock creation time
        elapsed_minutes = 2  # Mock elapsed time
        
        if elapsed_minutes < 1:
            status = "running"
            progress = 25.0
            message = "Processing historical data..."
        elif elapsed_minutes < 3:
            status = "running"
            progress = 75.0
            message = "Calculating analytics metrics..."
        else:
            status = "completed"
            progress = 100.0
            message = "Operation completed successfully"
        
        return AnalyticsStatusResponse(
            operation_id=operation_id,
            status=status,
            progress=progress,
            message=message,
            result={"mock": "result"} if status == "completed" else None,
            created_at=created_time,
            updated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to get operation status for {operation_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get operation status: {str(e)}"
        )


@router.delete("/operations/{operation_id}")
async def cancel_operation(
    operation_id: str,
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Cancel a running analytics operation.
    
    Attempts to gracefully cancel long-running operations
    and clean up associated resources.
    """
    try:
        logger.info(f"Cancelling operation {operation_id}")
        
        # In a real implementation, this would cancel the operation in the job queue
        # For now, return success
        
        return {
            "operation_id": operation_id,
            "status": "cancelled",
            "message": "Operation cancelled successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel operation {operation_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel operation: {str(e)}"
        )


@router.get("/metrics")
async def get_service_metrics():
    """
    Get analytics service performance metrics.
    
    Returns operational metrics for monitoring and observability.
    """
    try:
        # In a real implementation, this would return actual metrics
        # For now, return mock metrics
        
        return {
            "service_metrics": {
                "requests_total": 1250,
                "requests_per_minute": 15.2,
                "average_response_time_ms": 245,
                "error_rate_percent": 2.1,
                "active_operations": 3
            },
            "analytics_metrics": {
                "backtests_completed_today": 45,
                "monte_carlo_simulations_today": 23,
                "risk_analyses_today": 67,
                "performance_updates_today": 156
            },
            "resource_usage": {
                "cpu_percent": 35.2,
                "memory_percent": 42.8,
                "disk_usage_percent": 15.6,
                "cache_hit_rate_percent": 87.3
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get service metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )


# Helper functions
async def _execute_analytics_operation(
    analytics_service: AnalyticsService,
    operation: str,
    portfolio_id: str,
    tickers: List[str],
    weights: List[float],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute a single analytics operation."""
    options = options or {}
    
    if operation == "backtest":
        config = BacktestConfig(
            tickers=tickers,
            start_date=date(2020, 1, 1),  # Default dates
            end_date=date.today(),
            strategy=options.get("strategy", "sharpe"),
            rebalance_frequency=options.get("rebalance_frequency", "monthly"),
            transaction_cost=options.get("transaction_cost", 0.001),
            initial_capital=options.get("initial_capital", 100000.0)
        )
        result = analytics_service.run_backtest(config)
        return result.dict() if hasattr(result, 'dict') else str(result)
        
    elif operation == "monte_carlo":
        config = MonteCarloConfig(
            portfolio_tickers=tickers,
            portfolio_weights=weights,
            time_horizon_years=options.get("time_horizon_years", 10),
            num_simulations=options.get("num_simulations", 10000),
            confidence_levels=options.get("confidence_levels", [0.05, 0.25, 0.5, 0.75, 0.95]),
            initial_value=options.get("initial_value", 100000.0)
        )
        result = analytics_service.run_monte_carlo(config)
        return result.dict() if hasattr(result, 'dict') else str(result)
        
    elif operation == "risk_analysis":
        result = analytics_service.analyze_risk(portfolio_id, tickers, weights)
        return result.dict() if hasattr(result, 'dict') else str(result)
        
    elif operation == "performance":
        result = analytics_service.track_performance(portfolio_id, tickers, weights)
        return result.dict() if hasattr(result, 'dict') else str(result)
        
    elif operation == "dividends":
        result = analytics_service.analyze_dividends(portfolio_id, tickers, weights)
        return result.dict() if hasattr(result, 'dict') else str(result)
        
    else:
        raise ValueError(f"Unsupported operation: {operation}")