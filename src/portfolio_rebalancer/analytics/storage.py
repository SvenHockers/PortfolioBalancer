"""Analytics data storage interfaces and implementations."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import pandas as pd

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, DateTime, 
    Text, JSON, Boolean, Index, UniqueConstraint, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from .models import (
    BacktestResult, MonteCarloResult, RiskAnalysis, 
    PerformanceMetrics, DividendAnalysis, AnalyticsError
)

logger = logging.getLogger(__name__)

Base = declarative_base()


class BacktestResultTable(Base):
    """SQLAlchemy model for backtest results."""
    
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    config_hash = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    tickers = Column(JSON, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    strategy = Column(String(50), nullable=False)
    rebalance_frequency = Column(String(20), nullable=False)
    transaction_cost = Column(Float, nullable=False)
    initial_capital = Column(Float, nullable=False)
    
    # Results
    total_return = Column(Float)
    annualized_return = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    calmar_ratio = Column(Float)
    transaction_costs = Column(Float)
    num_rebalances = Column(Integer)
    final_value = Column(Float)
    
    # Serialized data
    returns_data = Column(JSON)
    allocation_data = Column(JSON)
    
    __table_args__ = (
        Index('idx_backtest_config', 'config_hash', 'created_at'),
        Index('idx_backtest_strategy', 'strategy', 'created_at'),
        Index('idx_backtest_dates', 'start_date', 'end_date'),
    )


class MonteCarloResultTable(Base):
    """SQLAlchemy model for Monte Carlo results."""
    
    __tablename__ = 'monte_carlo_results'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    portfolio_tickers = Column(JSON, nullable=False)
    portfolio_weights = Column(JSON, nullable=False)
    time_horizon_years = Column(Integer, nullable=False)
    num_simulations = Column(Integer, nullable=False)
    initial_value = Column(Float, nullable=False)
    
    # Results
    expected_value = Column(Float)
    probability_of_loss = Column(Float)
    value_at_risk_95 = Column(Float)
    conditional_var_95 = Column(Float)
    
    # Serialized data
    percentile_data = Column(JSON)
    simulation_summary = Column(JSON)
    
    __table_args__ = (
        Index('idx_monte_carlo_portfolio', 'portfolio_id', 'created_at'),
        Index('idx_monte_carlo_horizon', 'time_horizon_years', 'created_at'),
    )


class RiskAnalysisTable(Base):
    """SQLAlchemy model for risk analysis."""
    
    __tablename__ = 'risk_analysis'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(String(100), nullable=False)
    analysis_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Risk metrics
    portfolio_beta = Column(Float)
    tracking_error = Column(Float)
    information_ratio = Column(Float)
    var_95 = Column(Float)
    cvar_95 = Column(Float)
    max_drawdown = Column(Float)
    concentration_risk = Column(Float)
    
    # Serialized data
    correlation_data = Column(JSON)
    factor_exposures = Column(JSON)
    sector_exposures = Column(JSON)
    
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'analysis_date', name='uq_risk_portfolio_date'),
        Index('idx_risk_portfolio', 'portfolio_id', 'analysis_date'),
    )


class PerformanceMetricsTable(Base):
    """SQLAlchemy model for performance metrics."""
    
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(String(100), nullable=False)
    calculation_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Performance metrics
    total_return = Column(Float)
    annualized_return = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    alpha = Column(Float)
    beta = Column(Float)
    r_squared = Column(Float)
    tracking_error = Column(Float)
    information_ratio = Column(Float)
    
    # Serialized data
    performance_data = Column(JSON)
    
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'calculation_date', name='uq_perf_portfolio_date'),
        Index('idx_perf_portfolio', 'portfolio_id', 'calculation_date'),
    )


class DividendAnalysisTable(Base):
    """SQLAlchemy model for dividend analysis."""
    
    __tablename__ = 'dividend_analysis'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(String(100), nullable=False)
    analysis_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Dividend metrics
    current_yield = Column(Float)
    projected_annual_income = Column(Float)
    dividend_growth_rate = Column(Float)
    payout_ratio = Column(Float)
    dividend_coverage = Column(Float)
    income_sustainability_score = Column(Float)
    
    # Serialized data
    dividend_data = Column(JSON)
    top_contributors = Column(JSON)
    
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'analysis_date', name='uq_div_portfolio_date'),
        Index('idx_div_portfolio', 'portfolio_id', 'analysis_date'),
    )


class AnalyticsStorage(ABC):
    """Abstract interface for analytics data persistence."""
    
    @abstractmethod
    def store_backtest_result(self, result: BacktestResult) -> str:
        """Store backtest result and return ID."""
        pass
    
    @abstractmethod
    def get_backtest_result(self, result_id: str) -> Optional[BacktestResult]:
        """Retrieve backtest result by ID."""
        pass
    
    @abstractmethod
    def get_backtest_results_by_config_hash(self, config_hash: str) -> List[BacktestResult]:
        """Retrieve backtest results by configuration hash."""
        pass
    
    @abstractmethod
    def store_monte_carlo_result(self, result: MonteCarloResult) -> str:
        """Store Monte Carlo result and return ID."""
        pass
    
    @abstractmethod
    def get_monte_carlo_result(self, result_id: str) -> Optional[MonteCarloResult]:
        """Retrieve Monte Carlo result by ID."""
        pass
    
    @abstractmethod
    def store_risk_analysis(self, analysis: RiskAnalysis) -> None:
        """Store risk analysis (upsert by portfolio_id and date)."""
        pass
    
    @abstractmethod
    def get_risk_analysis(self, portfolio_id: str, analysis_date: date) -> Optional[RiskAnalysis]:
        """Retrieve risk analysis by portfolio and date."""
        pass
    
    @abstractmethod
    def store_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Store performance metrics (upsert by portfolio_id and date)."""
        pass
    
    @abstractmethod
    def get_performance_metrics(self, portfolio_id: str, calculation_date: date) -> Optional[PerformanceMetrics]:
        """Retrieve performance metrics by portfolio and date."""
        pass
    
    @abstractmethod
    def get_performance_history(self, portfolio_id: str, start_date: date, end_date: date) -> List[PerformanceMetrics]:
        """Retrieve performance metrics history for date range."""
        pass
    
    @abstractmethod
    def store_dividend_analysis(self, analysis: DividendAnalysis) -> None:
        """Store dividend analysis (upsert by portfolio_id and date)."""
        pass
    
    @abstractmethod
    def get_dividend_analysis(self, portfolio_id: str, analysis_date: date) -> Optional[DividendAnalysis]:
        """Retrieve dividend analysis by portfolio and date."""
        pass


class PostgreSQLAnalyticsStorage(AnalyticsStorage):
    """PostgreSQL implementation of analytics storage."""
    
    def __init__(self, database_url: str, pool_size: int = 10, max_overflow: int = 20):
        """
        Initialize PostgreSQL analytics storage.
        
        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size
            max_overflow: Maximum pool overflow
        """
        self.database_url = database_url
        
        # Create engine with connection pooling
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Validate connections before use
            pool_recycle=3600,   # Recycle connections after 1 hour
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        self._create_tables()
        
        logger.info(f"Initialized PostgreSQL analytics storage with pool_size={pool_size}")
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Analytics database tables created/verified")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create analytics tables: {e}")
            raise AnalyticsError(f"Database initialization failed: {e}")
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def store_backtest_result(self, result: BacktestResult) -> str:
        """Store backtest result and return ID."""
        try:
            with self.get_session() as session:
                # Create config hash for caching
                import hashlib
                config_str = f"{result.config.tickers}_{result.config.start_date}_{result.config.end_date}_{result.config.strategy}_{result.config.rebalance_frequency}_{result.config.transaction_cost}"
                config_hash = hashlib.sha256(config_str.encode()).hexdigest()
                
                db_result = BacktestResultTable(
                    config_hash=config_hash,
                    tickers=result.config.tickers,
                    start_date=result.config.start_date,
                    end_date=result.config.end_date,
                    strategy=result.config.strategy if isinstance(result.config.strategy, str) else result.config.strategy.value,
                    rebalance_frequency=result.config.rebalance_frequency if isinstance(result.config.rebalance_frequency, str) else result.config.rebalance_frequency.value,
                    transaction_cost=result.config.transaction_cost,
                    initial_capital=result.config.initial_capital,
                    total_return=result.total_return,
                    annualized_return=result.annualized_return,
                    volatility=result.volatility,
                    sharpe_ratio=result.sharpe_ratio,
                    max_drawdown=result.max_drawdown,
                    calmar_ratio=result.calmar_ratio,
                    transaction_costs=result.transaction_costs,
                    num_rebalances=result.num_rebalances,
                    final_value=result.final_value,
                    returns_data=result.returns_data,
                    allocation_data=result.allocation_data
                )
                
                session.add(db_result)
                session.flush()  # Get the ID
                
                result_id = str(db_result.id)
                logger.info(f"Stored backtest result with ID: {result_id}")
                return result_id
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to store backtest result: {e}")
            raise AnalyticsError(f"Failed to store backtest result: {e}")
    
    def get_backtest_result(self, result_id: str) -> Optional[BacktestResult]:
        """Retrieve backtest result by ID."""
        try:
            with self.get_session() as session:
                db_result = session.query(BacktestResultTable).filter(
                    BacktestResultTable.id == int(result_id)
                ).first()
                
                if not db_result:
                    return None
                
                return self._convert_backtest_result(db_result)
                
        except (SQLAlchemyError, ValueError) as e:
            logger.error(f"Failed to retrieve backtest result {result_id}: {e}")
            raise AnalyticsError(f"Failed to retrieve backtest result: {e}")
    
    def get_backtest_results_by_config_hash(self, config_hash: str) -> List[BacktestResult]:
        """Retrieve backtest results by configuration hash."""
        try:
            with self.get_session() as session:
                db_results = session.query(BacktestResultTable).filter(
                    BacktestResultTable.config_hash == config_hash
                ).order_by(BacktestResultTable.created_at.desc()).all()
                
                return [self._convert_backtest_result(db_result) for db_result in db_results]
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve backtest results by config hash: {e}")
            raise AnalyticsError(f"Failed to retrieve backtest results: {e}")
    
    def _convert_backtest_result(self, db_result: BacktestResultTable) -> BacktestResult:
        """Convert database result to BacktestResult model."""
        from .models import BacktestConfig, OptimizationStrategy, RebalanceFrequency
        
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
            total_return=db_result.total_return,
            annualized_return=db_result.annualized_return,
            volatility=db_result.volatility,
            sharpe_ratio=db_result.sharpe_ratio,
            max_drawdown=db_result.max_drawdown,
            calmar_ratio=db_result.calmar_ratio,
            transaction_costs=db_result.transaction_costs,
            num_rebalances=db_result.num_rebalances,
            final_value=db_result.final_value,
            returns_data=db_result.returns_data,
            allocation_data=db_result.allocation_data
        )
    
    def store_monte_carlo_result(self, result: MonteCarloResult) -> str:
        """Store Monte Carlo result and return ID."""
        try:
            with self.get_session() as session:
                db_result = MonteCarloResultTable(
                    portfolio_id=f"portfolio_{hash(str(result.config.portfolio_tickers))}",
                    portfolio_tickers=result.config.portfolio_tickers,
                    portfolio_weights=result.config.portfolio_weights,
                    time_horizon_years=result.config.time_horizon_years,
                    num_simulations=result.config.num_simulations,
                    initial_value=result.config.initial_value,
                    expected_value=result.expected_value,
                    probability_of_loss=result.probability_of_loss,
                    value_at_risk_95=result.value_at_risk_95,
                    conditional_var_95=result.conditional_var_95,
                    percentile_data=result.percentile_data,
                    simulation_summary=result.simulation_summary
                )
                
                session.add(db_result)
                session.flush()
                
                result_id = str(db_result.id)
                logger.info(f"Stored Monte Carlo result with ID: {result_id}")
                return result_id
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to store Monte Carlo result: {e}")
            raise AnalyticsError(f"Failed to store Monte Carlo result: {e}")
    
    def get_monte_carlo_result(self, result_id: str) -> Optional[MonteCarloResult]:
        """Retrieve Monte Carlo result by ID."""
        try:
            with self.get_session() as session:
                db_result = session.query(MonteCarloResultTable).filter(
                    MonteCarloResultTable.id == int(result_id)
                ).first()
                
                if not db_result:
                    return None
                
                return self._convert_monte_carlo_result(db_result)
                
        except (SQLAlchemyError, ValueError) as e:
            logger.error(f"Failed to retrieve Monte Carlo result {result_id}: {e}")
            raise AnalyticsError(f"Failed to retrieve Monte Carlo result: {e}")
    
    def _convert_monte_carlo_result(self, db_result: MonteCarloResultTable) -> MonteCarloResult:
        """Convert database result to MonteCarloResult model."""
        from .models import MonteCarloConfig
        
        config = MonteCarloConfig(
            portfolio_tickers=db_result.portfolio_tickers,
            portfolio_weights=db_result.portfolio_weights,
            time_horizon_years=db_result.time_horizon_years,
            num_simulations=db_result.num_simulations,
            initial_value=db_result.initial_value
        )
        
        return MonteCarloResult(
            config=config,
            expected_value=db_result.expected_value,
            probability_of_loss=db_result.probability_of_loss,
            value_at_risk_95=db_result.value_at_risk_95,
            conditional_var_95=db_result.conditional_var_95,
            percentile_data=db_result.percentile_data,
            simulation_summary=db_result.simulation_summary
        )
    
    def store_risk_analysis(self, analysis: RiskAnalysis) -> None:
        """Store risk analysis (upsert by portfolio_id and date)."""
        try:
            with self.get_session() as session:
                # Check if record exists
                existing = session.query(RiskAnalysisTable).filter(
                    RiskAnalysisTable.portfolio_id == analysis.portfolio_id,
                    RiskAnalysisTable.analysis_date == analysis.analysis_date
                ).first()
                
                if existing:
                    # Update existing record
                    existing.portfolio_beta = analysis.portfolio_beta
                    existing.tracking_error = analysis.tracking_error
                    existing.information_ratio = analysis.information_ratio
                    existing.var_95 = analysis.var_95
                    existing.cvar_95 = analysis.cvar_95
                    existing.max_drawdown = analysis.max_drawdown
                    existing.concentration_risk = analysis.concentration_risk
                    existing.correlation_data = analysis.correlation_data
                    existing.factor_exposures = analysis.factor_exposures
                    existing.sector_exposures = analysis.sector_exposures
                    existing.created_at = datetime.utcnow()
                else:
                    # Create new record
                    db_analysis = RiskAnalysisTable(
                        portfolio_id=analysis.portfolio_id,
                        analysis_date=analysis.analysis_date,
                        portfolio_beta=analysis.portfolio_beta,
                        tracking_error=analysis.tracking_error,
                        information_ratio=analysis.information_ratio,
                        var_95=analysis.var_95,
                        cvar_95=analysis.cvar_95,
                        max_drawdown=analysis.max_drawdown,
                        concentration_risk=analysis.concentration_risk,
                        correlation_data=analysis.correlation_data,
                        factor_exposures=analysis.factor_exposures,
                        sector_exposures=analysis.sector_exposures
                    )
                    session.add(db_analysis)
                
                logger.info(f"Stored risk analysis for portfolio {analysis.portfolio_id}")
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to store risk analysis: {e}")
            raise AnalyticsError(f"Failed to store risk analysis: {e}")
    
    def get_risk_analysis(self, portfolio_id: str, analysis_date: date) -> Optional[RiskAnalysis]:
        """Retrieve risk analysis by portfolio and date."""
        try:
            with self.get_session() as session:
                db_analysis = session.query(RiskAnalysisTable).filter(
                    RiskAnalysisTable.portfolio_id == portfolio_id,
                    RiskAnalysisTable.analysis_date == analysis_date
                ).first()
                
                if not db_analysis:
                    return None
                
                return RiskAnalysis(
                    portfolio_id=db_analysis.portfolio_id,
                    analysis_date=db_analysis.analysis_date,
                    portfolio_beta=db_analysis.portfolio_beta,
                    tracking_error=db_analysis.tracking_error,
                    information_ratio=db_analysis.information_ratio,
                    var_95=db_analysis.var_95,
                    cvar_95=db_analysis.cvar_95,
                    max_drawdown=db_analysis.max_drawdown,
                    concentration_risk=db_analysis.concentration_risk,
                    correlation_data=db_analysis.correlation_data,
                    factor_exposures=db_analysis.factor_exposures,
                    sector_exposures=db_analysis.sector_exposures
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve risk analysis: {e}")
            raise AnalyticsError(f"Failed to retrieve risk analysis: {e}")
    
    def store_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Store performance metrics (upsert by portfolio_id and date)."""
        try:
            with self.get_session() as session:
                # Check if record exists
                existing = session.query(PerformanceMetricsTable).filter(
                    PerformanceMetricsTable.portfolio_id == metrics.portfolio_id,
                    PerformanceMetricsTable.calculation_date == metrics.calculation_date
                ).first()
                
                if existing:
                    # Update existing record
                    existing.total_return = metrics.total_return
                    existing.annualized_return = metrics.annualized_return
                    existing.volatility = metrics.volatility
                    existing.sharpe_ratio = metrics.sharpe_ratio
                    existing.sortino_ratio = metrics.sortino_ratio
                    existing.alpha = metrics.alpha
                    existing.beta = metrics.beta
                    existing.r_squared = metrics.r_squared
                    existing.tracking_error = metrics.tracking_error
                    existing.information_ratio = metrics.information_ratio
                    existing.performance_data = metrics.performance_data
                    existing.created_at = datetime.utcnow()
                else:
                    # Create new record
                    db_metrics = PerformanceMetricsTable(
                        portfolio_id=metrics.portfolio_id,
                        calculation_date=metrics.calculation_date,
                        total_return=metrics.total_return,
                        annualized_return=metrics.annualized_return,
                        volatility=metrics.volatility,
                        sharpe_ratio=metrics.sharpe_ratio,
                        sortino_ratio=metrics.sortino_ratio,
                        alpha=metrics.alpha,
                        beta=metrics.beta,
                        r_squared=metrics.r_squared,
                        tracking_error=metrics.tracking_error,
                        information_ratio=metrics.information_ratio,
                        performance_data=metrics.performance_data
                    )
                    session.add(db_metrics)
                
                logger.info(f"Stored performance metrics for portfolio {metrics.portfolio_id}")
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to store performance metrics: {e}")
            raise AnalyticsError(f"Failed to store performance metrics: {e}")
    
    def get_performance_metrics(self, portfolio_id: str, calculation_date: date) -> Optional[PerformanceMetrics]:
        """Retrieve performance metrics by portfolio and date."""
        try:
            with self.get_session() as session:
                db_metrics = session.query(PerformanceMetricsTable).filter(
                    PerformanceMetricsTable.portfolio_id == portfolio_id,
                    PerformanceMetricsTable.calculation_date == calculation_date
                ).first()
                
                if not db_metrics:
                    return None
                
                return PerformanceMetrics(
                    portfolio_id=db_metrics.portfolio_id,
                    calculation_date=db_metrics.calculation_date,
                    total_return=db_metrics.total_return,
                    annualized_return=db_metrics.annualized_return,
                    volatility=db_metrics.volatility,
                    sharpe_ratio=db_metrics.sharpe_ratio,
                    sortino_ratio=db_metrics.sortino_ratio,
                    alpha=db_metrics.alpha,
                    beta=db_metrics.beta,
                    r_squared=db_metrics.r_squared,
                    tracking_error=db_metrics.tracking_error,
                    information_ratio=db_metrics.information_ratio,
                    performance_data=db_metrics.performance_data
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve performance metrics: {e}")
            raise AnalyticsError(f"Failed to retrieve performance metrics: {e}")
    
    def get_performance_history(self, portfolio_id: str, start_date: date, end_date: date) -> List[PerformanceMetrics]:
        """Retrieve performance metrics history for date range."""
        try:
            with self.get_session() as session:
                db_metrics_list = session.query(PerformanceMetricsTable).filter(
                    PerformanceMetricsTable.portfolio_id == portfolio_id,
                    PerformanceMetricsTable.calculation_date >= start_date,
                    PerformanceMetricsTable.calculation_date <= end_date
                ).order_by(PerformanceMetricsTable.calculation_date).all()
                
                return [
                    PerformanceMetrics(
                        portfolio_id=db_metrics.portfolio_id,
                        calculation_date=db_metrics.calculation_date,
                        total_return=db_metrics.total_return,
                        annualized_return=db_metrics.annualized_return,
                        volatility=db_metrics.volatility,
                        sharpe_ratio=db_metrics.sharpe_ratio,
                        sortino_ratio=db_metrics.sortino_ratio,
                        alpha=db_metrics.alpha,
                        beta=db_metrics.beta,
                        r_squared=db_metrics.r_squared,
                        tracking_error=db_metrics.tracking_error,
                        information_ratio=db_metrics.information_ratio,
                        performance_data=db_metrics.performance_data
                    )
                    for db_metrics in db_metrics_list
                ]
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve performance history: {e}")
            raise AnalyticsError(f"Failed to retrieve performance history: {e}")
    
    def store_dividend_analysis(self, analysis: DividendAnalysis) -> None:
        """Store dividend analysis (upsert by portfolio_id and date)."""
        try:
            with self.get_session() as session:
                # Check if record exists
                existing = session.query(DividendAnalysisTable).filter(
                    DividendAnalysisTable.portfolio_id == analysis.portfolio_id,
                    DividendAnalysisTable.analysis_date == analysis.analysis_date
                ).first()
                
                if existing:
                    # Update existing record
                    existing.current_yield = analysis.current_yield
                    existing.projected_annual_income = analysis.projected_annual_income
                    existing.dividend_growth_rate = analysis.dividend_growth_rate
                    existing.payout_ratio = analysis.payout_ratio
                    existing.dividend_coverage = analysis.dividend_coverage
                    existing.income_sustainability_score = analysis.income_sustainability_score
                    existing.dividend_data = analysis.dividend_data
                    existing.top_contributors = analysis.top_contributors
                    existing.created_at = datetime.utcnow()
                else:
                    # Create new record
                    db_analysis = DividendAnalysisTable(
                        portfolio_id=analysis.portfolio_id,
                        analysis_date=analysis.analysis_date,
                        current_yield=analysis.current_yield,
                        projected_annual_income=analysis.projected_annual_income,
                        dividend_growth_rate=analysis.dividend_growth_rate,
                        payout_ratio=analysis.payout_ratio,
                        dividend_coverage=analysis.dividend_coverage,
                        income_sustainability_score=analysis.income_sustainability_score,
                        dividend_data=analysis.dividend_data,
                        top_contributors=analysis.top_contributors
                    )
                    session.add(db_analysis)
                
                logger.info(f"Stored dividend analysis for portfolio {analysis.portfolio_id}")
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to store dividend analysis: {e}")
            raise AnalyticsError(f"Failed to store dividend analysis: {e}")
    
    def get_dividend_analysis(self, portfolio_id: str, analysis_date: date) -> Optional[DividendAnalysis]:
        """Retrieve dividend analysis by portfolio and date."""
        try:
            with self.get_session() as session:
                db_analysis = session.query(DividendAnalysisTable).filter(
                    DividendAnalysisTable.portfolio_id == portfolio_id,
                    DividendAnalysisTable.analysis_date == analysis_date
                ).first()
                
                if not db_analysis:
                    return None
                
                return DividendAnalysis(
                    portfolio_id=db_analysis.portfolio_id,
                    analysis_date=db_analysis.analysis_date,
                    current_yield=db_analysis.current_yield,
                    projected_annual_income=db_analysis.projected_annual_income,
                    dividend_growth_rate=db_analysis.dividend_growth_rate,
                    payout_ratio=db_analysis.payout_ratio,
                    dividend_coverage=db_analysis.dividend_coverage,
                    income_sustainability_score=db_analysis.income_sustainability_score,
                    dividend_data=db_analysis.dividend_data,
                    top_contributors=db_analysis.top_contributors
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve dividend analysis: {e}")
            raise AnalyticsError(f"Failed to retrieve dividend analysis: {e}")
    
    def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False