"""Database query optimization and data processing enhancements."""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import date, datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from sqlalchemy import (
    create_engine, text, Index, func, and_, or_, desc, asc,
    select, update, delete, insert
)
from sqlalchemy.orm import Session, sessionmaker, Query
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .storage import (
    BacktestResultTable, MonteCarloResultTable, RiskAnalysisTable,
    PerformanceMetricsTable, DividendAnalysisTable, Base
)
from .models import BacktestResult, MonteCarloResult, RiskAnalysis, PerformanceMetrics, DividendAnalysis
from .exceptions import AnalyticsError

logger = logging.getLogger(__name__)


@dataclass
class QueryPerformanceMetrics:
    """Query performance tracking."""
    query_type: str
    execution_time: float
    rows_affected: int
    cache_hit: bool = False
    optimization_applied: str = ""


class QueryOptimizer:
    """Database query optimization and performance enhancement."""
    
    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 40):
        """
        Initialize query optimizer.
        
        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size
            max_overflow: Maximum pool overflow
        """
        self.database_url = database_url
        self.performance_metrics: List[QueryPerformanceMetrics] = []
        
        # Enhanced connection pool configuration
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_timeout=30,
            echo=False,
            # PostgreSQL-specific optimizations
            connect_args={
                "options": "-c default_transaction_isolation=read_committed "
                          "-c statement_timeout=30000 "
                          "-c lock_timeout=10000"
            }
        )
        
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize optimizations
        self._create_optimized_indexes()
        self._configure_database_settings()
        
        logger.info("Query optimizer initialized with enhanced connection pooling")
    
    def _create_optimized_indexes(self):
        """Create optimized database indexes for common query patterns."""
        try:
            with self.engine.connect() as conn:
                # Composite indexes for common query patterns
                indexes = [
                    # Backtest result optimizations
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtest_strategy_dates "
                    "ON backtest_results (strategy, start_date, end_date, created_at DESC)",
                    
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtest_tickers_gin "
                    "ON backtest_results USING GIN (tickers)",
                    
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtest_performance "
                    "ON backtest_results (sharpe_ratio DESC, total_return DESC) "
                    "WHERE sharpe_ratio IS NOT NULL",
                    
                    # Performance metrics optimizations
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_perf_portfolio_date_range "
                    "ON performance_metrics (portfolio_id, calculation_date DESC, created_at DESC)",
                    
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_perf_metrics_composite "
                    "ON performance_metrics (portfolio_id, sharpe_ratio DESC, total_return DESC) "
                    "WHERE sharpe_ratio IS NOT NULL",
                    
                    # Risk analysis optimizations
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_portfolio_recent "
                    "ON risk_analysis (portfolio_id, analysis_date DESC, var_95) "
                    "WHERE var_95 IS NOT NULL",
                    
                    # Monte Carlo optimizations
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_monte_carlo_horizon_recent "
                    "ON monte_carlo_results (time_horizon_years, created_at DESC, expected_value) "
                    "WHERE expected_value IS NOT NULL",
                    
                    # Dividend analysis optimizations
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dividend_yield_recent "
                    "ON dividend_analysis (portfolio_id, analysis_date DESC, current_yield) "
                    "WHERE current_yield IS NOT NULL"
                ]
                
                for index_sql in indexes:
                    try:
                        conn.execute(text(index_sql))
                        logger.debug(f"Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
                    except Exception as e:
                        # Index might already exist or creation failed
                        logger.debug(f"Index creation skipped: {e}")
                
                conn.commit()
                logger.info("Optimized database indexes created/verified")
                
        except SQLAlchemyError as e:
            logger.warning(f"Failed to create optimized indexes: {e}")
    
    def _configure_database_settings(self):
        """Configure PostgreSQL settings for optimal performance."""
        try:
            with self.engine.connect() as conn:
                # Optimize for analytics workload
                settings = [
                    "SET work_mem = '256MB'",  # Increase work memory for complex queries
                    "SET maintenance_work_mem = '512MB'",  # For index operations
                    "SET effective_cache_size = '2GB'",  # Assume reasonable cache size
                    "SET random_page_cost = 1.1",  # SSD-optimized
                    "SET seq_page_cost = 1.0",
                    "SET cpu_tuple_cost = 0.01",
                    "SET cpu_index_tuple_cost = 0.005",
                    "SET cpu_operator_cost = 0.0025"
                ]
                
                for setting in settings:
                    try:
                        conn.execute(text(setting))
                    except Exception as e:
                        logger.debug(f"Setting configuration skipped: {e}")
                
                logger.info("Database settings optimized for analytics workload")
                
        except SQLAlchemyError as e:
            logger.warning(f"Failed to configure database settings: {e}")
    
    @contextmanager
    def get_optimized_session(self):
        """Get database session with performance monitoring."""
        session = self.SessionLocal()
        start_time = time.time()
        
        try:
            yield session
            session.commit()
            
            execution_time = time.time() - start_time
            if execution_time > 1.0:  # Log slow sessions
                logger.warning(f"Slow database session: {execution_time:.2f}s")
                
        except Exception as e:
            session.rollback()
            execution_time = time.time() - start_time
            logger.error(f"Database session error after {execution_time:.2f}s: {e}")
            raise
        finally:
            session.close()
    
    def _track_query_performance(self, query_type: str, execution_time: float, 
                               rows_affected: int, optimization: str = ""):
        """Track query performance metrics."""
        metric = QueryPerformanceMetrics(
            query_type=query_type,
            execution_time=execution_time,
            rows_affected=rows_affected,
            optimization_applied=optimization
        )
        
        self.performance_metrics.append(metric)
        
        # Keep only recent metrics (last 1000)
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
        
        # Log slow queries
        if execution_time > 2.0:
            logger.warning(f"Slow query detected: {query_type} took {execution_time:.2f}s")
    
    def get_backtest_results_optimized(self, 
                                     strategy: Optional[str] = None,
                                     start_date: Optional[date] = None,
                                     end_date: Optional[date] = None,
                                     tickers: Optional[List[str]] = None,
                                     limit: int = 100,
                                     order_by: str = "created_at") -> List[BacktestResult]:
        """
        Optimized backtest results retrieval with intelligent filtering.
        
        Args:
            strategy: Filter by optimization strategy
            start_date: Filter by backtest start date
            end_date: Filter by backtest end date
            tickers: Filter by ticker symbols
            limit: Maximum number of results
            order_by: Ordering field (created_at, sharpe_ratio, total_return)
            
        Returns:
            List of backtest results
        """
        start_time = time.time()
        
        try:
            with self.get_optimized_session() as session:
                # Build optimized query
                query = session.query(BacktestResultTable)
                
                # Apply filters with index-friendly conditions
                if strategy:
                    query = query.filter(BacktestResultTable.strategy == strategy)
                
                if start_date:
                    query = query.filter(BacktestResultTable.start_date >= start_date)
                
                if end_date:
                    query = query.filter(BacktestResultTable.end_date <= end_date)
                
                if tickers:
                    # Use GIN index for array containment
                    query = query.filter(BacktestResultTable.tickers.op('@>')([tickers]))
                
                # Optimize ordering
                if order_by == "sharpe_ratio":
                    query = query.filter(BacktestResultTable.sharpe_ratio.isnot(None))
                    query = query.order_by(desc(BacktestResultTable.sharpe_ratio))
                elif order_by == "total_return":
                    query = query.filter(BacktestResultTable.total_return.isnot(None))
                    query = query.order_by(desc(BacktestResultTable.total_return))
                else:
                    query = query.order_by(desc(BacktestResultTable.created_at))
                
                # Apply limit
                query = query.limit(limit)
                
                # Execute query
                db_results = query.all()
                
                # Convert to domain models
                results = []
                for db_result in db_results:
                    try:
                        result = self._convert_backtest_result_optimized(db_result)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to convert backtest result {db_result.id}: {e}")
                        continue
                
                execution_time = time.time() - start_time
                self._track_query_performance(
                    "get_backtest_results_optimized", 
                    execution_time, 
                    len(results),
                    f"filtered_by_{strategy or 'all'}_ordered_by_{order_by}"
                )
                
                return results
                
        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            self._track_query_performance("get_backtest_results_optimized", execution_time, 0)
            logger.error(f"Optimized backtest query failed: {e}")
            raise AnalyticsError(f"Failed to retrieve backtest results: {e}")
    
    def get_performance_history_optimized(self, 
                                        portfolio_id: str,
                                        start_date: date,
                                        end_date: date,
                                        metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Optimized performance history retrieval with selective column loading.
        
        Args:
            portfolio_id: Portfolio identifier
            start_date: Start date for history
            end_date: End date for history
            metrics: Specific metrics to retrieve (optional)
            
        Returns:
            DataFrame with performance history
        """
        start_time = time.time()
        
        try:
            with self.get_optimized_session() as session:
                # Build selective query to reduce data transfer
                if metrics:
                    # Only select requested metrics
                    columns = [PerformanceMetricsTable.calculation_date]
                    for metric in metrics:
                        if hasattr(PerformanceMetricsTable, metric):
                            columns.append(getattr(PerformanceMetricsTable, metric))
                    
                    query = session.query(*columns)
                else:
                    query = session.query(PerformanceMetricsTable)
                
                # Apply optimized filters
                query = query.filter(
                    and_(
                        PerformanceMetricsTable.portfolio_id == portfolio_id,
                        PerformanceMetricsTable.calculation_date >= start_date,
                        PerformanceMetricsTable.calculation_date <= end_date
                    )
                ).order_by(PerformanceMetricsTable.calculation_date)
                
                # Execute query and convert to DataFrame
                if metrics:
                    results = query.all()
                    data = []
                    for result in results:
                        row = {'calculation_date': result[0]}
                        for i, metric in enumerate(metrics, 1):
                            row[metric] = result[i] if i < len(result) else None
                        data.append(row)
                    
                    df = pd.DataFrame(data)
                else:
                    # Full object query
                    db_results = query.all()
                    data = []
                    for db_result in db_results:
                        data.append({
                            'calculation_date': db_result.calculation_date,
                            'total_return': db_result.total_return,
                            'annualized_return': db_result.annualized_return,
                            'volatility': db_result.volatility,
                            'sharpe_ratio': db_result.sharpe_ratio,
                            'sortino_ratio': db_result.sortino_ratio,
                            'alpha': db_result.alpha,
                            'beta': db_result.beta,
                            'r_squared': db_result.r_squared,
                            'tracking_error': db_result.tracking_error,
                            'information_ratio': db_result.information_ratio
                        })
                    
                    df = pd.DataFrame(data)
                
                # Set date as index for time series operations
                if not df.empty:
                    df.set_index('calculation_date', inplace=True)
                
                execution_time = time.time() - start_time
                self._track_query_performance(
                    "get_performance_history_optimized",
                    execution_time,
                    len(df),
                    f"selective_columns_{len(metrics) if metrics else 'all'}"
                )
                
                return df
                
        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            self._track_query_performance("get_performance_history_optimized", execution_time, 0)
            logger.error(f"Optimized performance history query failed: {e}")
            raise AnalyticsError(f"Failed to retrieve performance history: {e}")
    
    def bulk_upsert_performance_metrics(self, metrics_list: List[PerformanceMetrics]) -> int:
        """
        Bulk upsert performance metrics using PostgreSQL-specific optimizations.
        
        Args:
            metrics_list: List of performance metrics to upsert
            
        Returns:
            Number of records affected
        """
        if not metrics_list:
            return 0
        
        start_time = time.time()
        
        try:
            with self.get_optimized_session() as session:
                # Prepare data for bulk upsert
                data = []
                for metrics in metrics_list:
                    data.append({
                        'portfolio_id': metrics.portfolio_id,
                        'calculation_date': metrics.calculation_date,
                        'total_return': metrics.total_return,
                        'annualized_return': metrics.annualized_return,
                        'volatility': metrics.volatility,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'sortino_ratio': metrics.sortino_ratio,
                        'alpha': metrics.alpha,
                        'beta': metrics.beta,
                        'r_squared': metrics.r_squared,
                        'tracking_error': metrics.tracking_error,
                        'information_ratio': metrics.information_ratio,
                        'performance_data': metrics.performance_data,
                        'created_at': datetime.utcnow()
                    })
                
                # Use PostgreSQL UPSERT (ON CONFLICT DO UPDATE)
                stmt = pg_insert(PerformanceMetricsTable).values(data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['portfolio_id', 'calculation_date'],
                    set_={
                        'total_return': stmt.excluded.total_return,
                        'annualized_return': stmt.excluded.annualized_return,
                        'volatility': stmt.excluded.volatility,
                        'sharpe_ratio': stmt.excluded.sharpe_ratio,
                        'sortino_ratio': stmt.excluded.sortino_ratio,
                        'alpha': stmt.excluded.alpha,
                        'beta': stmt.excluded.beta,
                        'r_squared': stmt.excluded.r_squared,
                        'tracking_error': stmt.excluded.tracking_error,
                        'information_ratio': stmt.excluded.information_ratio,
                        'performance_data': stmt.excluded.performance_data,
                        'created_at': stmt.excluded.created_at
                    }
                )
                
                result = session.execute(stmt)
                rows_affected = result.rowcount
                
                execution_time = time.time() - start_time
                self._track_query_performance(
                    "bulk_upsert_performance_metrics",
                    execution_time,
                    rows_affected,
                    f"batch_size_{len(metrics_list)}"
                )
                
                logger.info(f"Bulk upserted {rows_affected} performance metrics records")
                return rows_affected
                
        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            self._track_query_performance("bulk_upsert_performance_metrics", execution_time, 0)
            logger.error(f"Bulk upsert failed: {e}")
            raise AnalyticsError(f"Failed to bulk upsert performance metrics: {e}")
    
    def get_aggregated_portfolio_stats(self, 
                                     portfolio_ids: List[str],
                                     start_date: date,
                                     end_date: date) -> Dict[str, Dict[str, float]]:
        """
        Get aggregated statistics for multiple portfolios efficiently.
        
        Args:
            portfolio_ids: List of portfolio identifiers
            start_date: Start date for aggregation
            end_date: End date for aggregation
            
        Returns:
            Dictionary with aggregated stats by portfolio
        """
        start_time = time.time()
        
        try:
            with self.get_optimized_session() as session:
                # Single query to get aggregated stats for all portfolios
                query = session.query(
                    PerformanceMetricsTable.portfolio_id,
                    func.avg(PerformanceMetricsTable.total_return).label('avg_return'),
                    func.stddev(PerformanceMetricsTable.total_return).label('return_volatility'),
                    func.avg(PerformanceMetricsTable.sharpe_ratio).label('avg_sharpe'),
                    func.max(PerformanceMetricsTable.sharpe_ratio).label('max_sharpe'),
                    func.min(PerformanceMetricsTable.sharpe_ratio).label('min_sharpe'),
                    func.count().label('data_points')
                ).filter(
                    and_(
                        PerformanceMetricsTable.portfolio_id.in_(portfolio_ids),
                        PerformanceMetricsTable.calculation_date >= start_date,
                        PerformanceMetricsTable.calculation_date <= end_date,
                        PerformanceMetricsTable.total_return.isnot(None),
                        PerformanceMetricsTable.sharpe_ratio.isnot(None)
                    )
                ).group_by(PerformanceMetricsTable.portfolio_id)
                
                results = query.all()
                
                # Convert to dictionary
                stats = {}
                for result in results:
                    stats[result.portfolio_id] = {
                        'avg_return': float(result.avg_return or 0),
                        'return_volatility': float(result.return_volatility or 0),
                        'avg_sharpe': float(result.avg_sharpe or 0),
                        'max_sharpe': float(result.max_sharpe or 0),
                        'min_sharpe': float(result.min_sharpe or 0),
                        'data_points': int(result.data_points or 0)
                    }
                
                execution_time = time.time() - start_time
                self._track_query_performance(
                    "get_aggregated_portfolio_stats",
                    execution_time,
                    len(results),
                    f"portfolios_{len(portfolio_ids)}"
                )
                
                return stats
                
        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            self._track_query_performance("get_aggregated_portfolio_stats", execution_time, 0)
            logger.error(f"Aggregated stats query failed: {e}")
            raise AnalyticsError(f"Failed to get aggregated portfolio stats: {e}")
    
    def _convert_backtest_result_optimized(self, db_result: BacktestResultTable) -> BacktestResult:
        """Optimized conversion of database result to domain model."""
        from .models import BacktestConfig
        
        # Lazy loading of complex data
        config = BacktestConfig(
            tickers=db_result.tickers,
            start_date=db_result.start_date,
            end_date=db_result.end_date,
            strategy=db_result.strategy,
            rebalance_frequency=db_result.rebalance_frequency,
            transaction_cost=db_result.transaction_cost,
            initial_capital=db_result.initial_capital
        )
        
        return BacktestResult(
            config=config,
            total_return=db_result.total_return or 0.0,
            annualized_return=db_result.annualized_return or 0.0,
            volatility=db_result.volatility or 0.0,
            sharpe_ratio=db_result.sharpe_ratio or 0.0,
            max_drawdown=db_result.max_drawdown or 0.0,
            calmar_ratio=db_result.calmar_ratio or 0.0,
            transaction_costs=db_result.transaction_costs or 0.0,
            num_rebalances=db_result.num_rebalances or 0,
            final_value=db_result.final_value or db_result.initial_capital,
            returns_data=db_result.returns_data,
            allocation_data=db_result.allocation_data
        )
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Clean up old analytics data to maintain performance.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Dictionary with cleanup statistics
        """
        start_time = time.time()
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        cleanup_stats = {}
        
        try:
            with self.get_optimized_session() as session:
                # Clean up old backtest results (keep recent ones)
                backtest_deleted = session.query(BacktestResultTable).filter(
                    BacktestResultTable.created_at < cutoff_date
                ).delete()
                
                # Clean up old Monte Carlo results
                monte_carlo_deleted = session.query(MonteCarloResultTable).filter(
                    MonteCarloResultTable.created_at < cutoff_date
                ).delete()
                
                # Keep performance metrics longer (they're smaller)
                perf_cutoff = datetime.utcnow() - timedelta(days=days_to_keep * 2)
                performance_deleted = session.query(PerformanceMetricsTable).filter(
                    PerformanceMetricsTable.created_at < perf_cutoff
                ).delete()
                
                cleanup_stats = {
                    'backtest_results_deleted': backtest_deleted,
                    'monte_carlo_results_deleted': monte_carlo_deleted,
                    'performance_metrics_deleted': performance_deleted,
                    'cleanup_date': cutoff_date.isoformat()
                }
                
                execution_time = time.time() - start_time
                total_deleted = sum(cleanup_stats.values() if isinstance(v, int) else 0 for v in cleanup_stats.values())
                
                self._track_query_performance(
                    "cleanup_old_data",
                    execution_time,
                    total_deleted,
                    f"days_kept_{days_to_keep}"
                )
                
                logger.info(f"Cleaned up {total_deleted} old records in {execution_time:.2f}s")
                return cleanup_stats
                
        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            self._track_query_performance("cleanup_old_data", execution_time, 0)
            logger.error(f"Data cleanup failed: {e}")
            raise AnalyticsError(f"Failed to cleanup old data: {e}")
    
    def get_query_performance_report(self) -> Dict[str, Any]:
        """
        Generate query performance report.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.performance_metrics:
            return {'message': 'No performance data available'}
        
        # Calculate statistics
        total_queries = len(self.performance_metrics)
        avg_execution_time = sum(m.execution_time for m in self.performance_metrics) / total_queries
        slow_queries = [m for m in self.performance_metrics if m.execution_time > 1.0]
        
        # Group by query type
        by_type = {}
        for metric in self.performance_metrics:
            if metric.query_type not in by_type:
                by_type[metric.query_type] = []
            by_type[metric.query_type].append(metric)
        
        type_stats = {}
        for query_type, metrics in by_type.items():
            type_stats[query_type] = {
                'count': len(metrics),
                'avg_time': sum(m.execution_time for m in metrics) / len(metrics),
                'max_time': max(m.execution_time for m in metrics),
                'total_rows': sum(m.rows_affected for m in metrics)
            }
        
        return {
            'total_queries': total_queries,
            'avg_execution_time': avg_execution_time,
            'slow_queries_count': len(slow_queries),
            'slow_query_threshold': 1.0,
            'by_query_type': type_stats,
            'recent_slow_queries': [
                {
                    'type': m.query_type,
                    'time': m.execution_time,
                    'rows': m.rows_affected,
                    'optimization': m.optimization_applied
                }
                for m in slow_queries[-10:]  # Last 10 slow queries
            ]
        }
    
    def vacuum_analyze_tables(self):
        """Run VACUUM ANALYZE on analytics tables for optimal performance."""
        try:
            with self.engine.connect() as conn:
                # Run VACUUM ANALYZE on each table
                tables = [
                    'backtest_results',
                    'monte_carlo_results',
                    'risk_analysis',
                    'performance_metrics',
                    'dividend_analysis'
                ]
                
                for table in tables:
                    try:
                        conn.execute(text(f"VACUUM ANALYZE {table}"))
                        logger.info(f"VACUUM ANALYZE completed for {table}")
                    except Exception as e:
                        logger.warning(f"VACUUM ANALYZE failed for {table}: {e}")
                
                conn.commit()
                logger.info("Database maintenance completed")
                
        except SQLAlchemyError as e:
            logger.error(f"Database maintenance failed: {e}")