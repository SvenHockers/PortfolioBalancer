"""Monte Carlo simulation engine for portfolio projections."""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from dataclasses import dataclass

from ..models import (
    MonteCarloConfig, 
    MonteCarloResult, 
    StressTestResult, 
    VaRResult
)
from ..exceptions import SimulationError, InsufficientDataError
from ...common.models import Portfolio


logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Definition of a stress testing scenario."""
    name: str
    description: str
    return_shocks: Dict[str, float]  # ticker -> return shock
    correlation_multiplier: float = 1.0
    duration_days: int = 252  # 1 year default
    volatility_multiplier: float = 1.0  # Volatility shock multiplier


@dataclass
class HistoricalCrisisScenario:
    """Historical crisis scenario definition."""
    name: str
    start_date: str
    end_date: str
    description: str


class MonteCarloEngine:
    """Monte Carlo simulation engine for portfolio projections."""
    
    def __init__(self, data_storage):
        """Initialize Monte Carlo engine.
        
        Args:
            data_storage: Data storage interface for historical data
        """
        self.data_storage = data_storage
        self.logger = logging.getLogger(__name__)
        
    def run_simulation(self, config: MonteCarloConfig) -> MonteCarloResult:
        """Run Monte Carlo portfolio projection simulation.
        
        Args:
            config: Monte Carlo simulation configuration
            
        Returns:
            MonteCarloResult with simulation results
            
        Raises:
            SimulationError: If simulation fails
            InsufficientDataError: If insufficient historical data
        """
        try:
            self.logger.info(f"Starting Monte Carlo simulation with {config.num_simulations} simulations")
            
            # Get historical data for return distribution estimation
            historical_data = self._get_historical_data(config.portfolio_tickers)
            
            # Calculate return statistics
            return_stats = self._calculate_return_statistics(historical_data, config.portfolio_weights)
            
            # Run simulations
            simulation_paths = self._generate_simulation_paths(config, return_stats)
            
            # Calculate results
            results = self._calculate_simulation_results(config, simulation_paths)
            
            self.logger.info("Monte Carlo simulation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation failed: {str(e)}")
            raise SimulationError(f"Simulation failed: {str(e)}")
    
    def stress_test(self, portfolio: Portfolio, scenarios: List[StressScenario]) -> StressTestResult:
        """Run stress testing under various market scenarios.
        
        Args:
            portfolio: Portfolio to stress test
            scenarios: List of stress scenarios to test
            
        Returns:
            StressTestResult with stress test results
        """
        try:
            self.logger.info(f"Running stress test with {len(scenarios)} scenarios")
            
            results = {}
            worst_case_loss = 0.0
            
            for scenario in scenarios:
                loss = self._calculate_scenario_loss(portfolio, scenario)
                results[scenario.name] = loss
                worst_case_loss = min(worst_case_loss, loss)
            
            # Estimate recovery time based on historical volatility
            recovery_time = self._estimate_recovery_time(portfolio, abs(worst_case_loss))
            
            return StressTestResult(
                portfolio_id=portfolio.id,
                test_date=date.today(),
                scenarios=[s.name for s in scenarios],
                results=results,
                worst_case_loss=worst_case_loss,
                recovery_time_estimate=recovery_time
            )
            
        except Exception as e:
            self.logger.error(f"Stress test failed: {str(e)}")
            raise SimulationError(f"Stress test failed: {str(e)}")
    
    def calculate_var(self, 
                     portfolio: Portfolio,
                     confidence_level: float = 0.05,
                     time_horizon_days: int = 252,
                     method: str = "historical_simulation") -> VaRResult:
        """Calculate Value at Risk and Conditional VaR.
        
        Args:
            portfolio: Portfolio to analyze
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            time_horizon_days: Time horizon in days
            method: VaR calculation method ('historical_simulation', 'parametric', 'monte_carlo')
            
        Returns:
            VaRResult with VaR calculations
        """
        try:
            self.logger.info(f"Calculating VaR at {(1-confidence_level)*100}% confidence using {method}")
            
            if method == "historical_simulation":
                return self._calculate_historical_var(portfolio, confidence_level, time_horizon_days)
            elif method == "parametric":
                return self._calculate_parametric_var(portfolio, confidence_level, time_horizon_days)
            elif method == "monte_carlo":
                return self._calculate_monte_carlo_var(portfolio, confidence_level, time_horizon_days)
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {str(e)}")
            raise SimulationError(f"VaR calculation failed: {str(e)}")
    
    def stress_test_historical_crises(self, portfolio: Portfolio) -> StressTestResult:
        """Run stress test using historical crisis scenarios.
        
        Args:
            portfolio: Portfolio to stress test
            
        Returns:
            StressTestResult with historical crisis results
        """
        try:
            self.logger.info("Running historical crisis stress test")
            
            # Define major historical crisis periods
            crisis_scenarios = [
                HistoricalCrisisScenario(
                    name="2008 Financial Crisis",
                    start_date="2007-10-01",
                    end_date="2009-03-31",
                    description="Global financial crisis and market crash"
                ),
                HistoricalCrisisScenario(
                    name="COVID-19 Crash",
                    start_date="2020-02-01",
                    end_date="2020-04-30",
                    description="COVID-19 pandemic market crash"
                ),
                HistoricalCrisisScenario(
                    name="Dot-com Crash",
                    start_date="2000-03-01",
                    end_date="2002-10-31",
                    description="Technology bubble burst"
                ),
                HistoricalCrisisScenario(
                    name="European Debt Crisis",
                    start_date="2010-05-01",
                    end_date="2012-07-31",
                    description="European sovereign debt crisis"
                )
            ]
            
            results = {}
            worst_case_loss = 0.0
            
            for scenario in crisis_scenarios:
                try:
                    loss = self._calculate_historical_crisis_loss(portfolio, scenario)
                    results[scenario.name] = loss
                    worst_case_loss = min(worst_case_loss, loss)
                except Exception as e:
                    self.logger.warning(f"Could not calculate loss for {scenario.name}: {str(e)}")
                    results[scenario.name] = 0.0
            
            # Estimate recovery time based on worst case
            recovery_time = self._estimate_recovery_time(portfolio, abs(worst_case_loss))
            
            return StressTestResult(
                portfolio_id=portfolio.id,
                test_date=date.today(),
                scenarios=list(results.keys()),
                results=results,
                worst_case_loss=worst_case_loss,
                recovery_time_estimate=recovery_time
            )
            
        except Exception as e:
            self.logger.error(f"Historical crisis stress test failed: {str(e)}")
            raise SimulationError(f"Historical crisis stress test failed: {str(e)}")
    
    def calculate_tail_risk_metrics(self, portfolio: Portfolio) -> Dict[str, float]:
        """Calculate comprehensive tail risk metrics.
        
        Args:
            portfolio: Portfolio to analyze
            
        Returns:
            Dictionary with tail risk metrics
        """
        try:
            self.logger.info("Calculating tail risk metrics")
            
            # Get historical returns
            historical_data = self._get_historical_data(portfolio.tickers)
            portfolio_returns = self._calculate_portfolio_returns(historical_data, portfolio.weights)
            
            # Calculate various tail risk metrics
            metrics = {}
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = drawdowns.min()
            
            # Expected shortfall (CVaR) at different confidence levels
            for confidence in [0.01, 0.05, 0.10]:
                var_level = np.percentile(portfolio_returns, confidence * 100)
                cvar = portfolio_returns[portfolio_returns <= var_level].mean()
                metrics[f'expected_shortfall_{int((1-confidence)*100)}'] = cvar
            
            # Tail ratio (average of top 5% returns / average of bottom 5% returns)
            top_5_pct = portfolio_returns.quantile(0.95)
            bottom_5_pct = portfolio_returns.quantile(0.05)
            top_returns = portfolio_returns[portfolio_returns >= top_5_pct].mean()
            bottom_returns = portfolio_returns[portfolio_returns <= bottom_5_pct].mean()
            metrics['tail_ratio'] = abs(top_returns / bottom_returns) if bottom_returns != 0 else np.inf
            
            # Skewness and kurtosis
            metrics['skewness'] = stats.skew(portfolio_returns)
            metrics['excess_kurtosis'] = stats.kurtosis(portfolio_returns)
            
            # Downside deviation
            negative_returns = portfolio_returns[portfolio_returns < 0]
            metrics['downside_deviation'] = negative_returns.std() if len(negative_returns) > 0 else 0.0
            
            # Pain index (average drawdown)
            metrics['pain_index'] = drawdowns.mean()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Tail risk calculation failed: {str(e)}")
            raise SimulationError(f"Tail risk calculation failed: {str(e)}")
    
    def run_correlation_aware_simulation(self, 
                                       config: MonteCarloConfig,
                                       correlation_shock: float = 1.0) -> MonteCarloResult:
        """Run Monte Carlo simulation with correlation shock scenarios.
        
        Args:
            config: Monte Carlo configuration
            correlation_shock: Multiplier for correlation matrix (1.0 = normal, >1.0 = higher correlation)
            
        Returns:
            MonteCarloResult with correlation-adjusted simulation
        """
        try:
            self.logger.info(f"Running correlation-aware simulation with shock factor {correlation_shock}")
            
            # Get historical data and calculate statistics
            historical_data = self._get_historical_data(config.portfolio_tickers)
            return_stats = self._calculate_return_statistics(historical_data, config.portfolio_weights)
            
            # Apply correlation shock
            original_corr = return_stats['correlation_matrix']
            shocked_corr = self._apply_correlation_shock(original_corr, correlation_shock)
            return_stats['correlation_matrix'] = shocked_corr
            
            # Generate simulation paths with shocked correlations
            simulation_paths = self._generate_simulation_paths(config, return_stats)
            
            # Calculate results
            results = self._calculate_simulation_results(config, simulation_paths)
            
            self.logger.info("Correlation-aware simulation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Correlation-aware simulation failed: {str(e)}")
            raise SimulationError(f"Correlation-aware simulation failed: {str(e)}")
    
    def _get_historical_data(self, tickers: List[str]) -> pd.DataFrame:
        """Get historical price data for tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            InsufficientDataError: If insufficient data available
        """
        try:
            # Get data from storage (assuming it returns price data)
            data_frames = []
            
            for ticker in tickers:
                ticker_data = self.data_storage.get_price_data(ticker)
                if ticker_data is None or len(ticker_data) < 252:  # Need at least 1 year
                    raise InsufficientDataError(
                        f"Insufficient data for {ticker}: need at least 252 days"
                    )
                
                # Ensure we have 'close' column and rename to ticker
                if 'close' in ticker_data.columns:
                    ticker_series = ticker_data['close'].rename(ticker)
                elif 'Close' in ticker_data.columns:
                    ticker_series = ticker_data['Close'].rename(ticker)
                else:
                    # Assume first numeric column is price
                    numeric_cols = ticker_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) == 0:
                        raise InsufficientDataError(f"No price data found for {ticker}")
                    ticker_series = ticker_data[numeric_cols[0]].rename(ticker)
                
                data_frames.append(ticker_series)
            
            # Combine all ticker data
            combined_data = pd.concat(data_frames, axis=1)
            combined_data = combined_data.dropna()
            
            if len(combined_data) < 252:
                raise InsufficientDataError(
                    f"Insufficient overlapping data: {len(combined_data)} days available, need at least 252"
                )
            
            return combined_data
            
        except Exception as e:
            if isinstance(e, InsufficientDataError):
                raise
            raise SimulationError(f"Failed to get historical data: {str(e)}")
    
    def _calculate_return_statistics(self, 
                                   price_data: pd.DataFrame, 
                                   weights: List[float]) -> Dict[str, Any]:
        """Calculate return statistics from historical data.
        
        Args:
            price_data: Historical price data
            weights: Portfolio weights
            
        Returns:
            Dictionary with return statistics
        """
        # Calculate daily returns
        returns = price_data.pct_change().dropna()
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate statistics
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Calculate correlation matrix for individual assets
        correlation_matrix = returns.corr().values
        
        # Individual asset statistics
        individual_means = returns.mean().values
        individual_stds = returns.std().values
        
        return {
            'portfolio_mean': mean_return,
            'portfolio_std': std_return,
            'individual_means': individual_means,
            'individual_stds': individual_stds,
            'correlation_matrix': correlation_matrix,
            'weights': np.array(weights),
            'historical_returns': portfolio_returns
        }
    
    def _generate_simulation_paths(self, 
                                 config: MonteCarloConfig, 
                                 return_stats: Dict[str, Any]) -> np.ndarray:
        """Generate Monte Carlo simulation paths.
        
        Args:
            config: Monte Carlo configuration
            return_stats: Return statistics from historical data
            
        Returns:
            Array of simulation paths [num_simulations, time_steps]
        """
        num_days = config.time_horizon_years * 252  # Trading days per year
        num_assets = len(config.portfolio_tickers)
        
        # Generate correlated random returns for individual assets
        np.random.seed(42)  # For reproducibility
        
        # Generate random normal returns
        random_returns = np.random.multivariate_normal(
            mean=return_stats['individual_means'],
            cov=np.outer(return_stats['individual_stds'], return_stats['individual_stds']) * 
                return_stats['correlation_matrix'],
            size=(config.num_simulations, num_days)
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = np.tensordot(random_returns, return_stats['weights'], axes=([2], [0]))
        
        # Convert returns to cumulative values
        initial_value = config.initial_value
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)
        simulation_paths = initial_value * cumulative_returns
        
        return simulation_paths
    
    def _calculate_simulation_results(self, 
                                    config: MonteCarloConfig, 
                                    simulation_paths: np.ndarray) -> MonteCarloResult:
        """Calculate results from simulation paths.
        
        Args:
            config: Monte Carlo configuration
            simulation_paths: Array of simulation paths
            
        Returns:
            MonteCarloResult with calculated results
        """
        # Final values from all simulations
        final_values = simulation_paths[:, -1]
        
        # Calculate key metrics
        expected_value = np.mean(final_values)
        probability_of_loss = np.mean(final_values < config.initial_value)
        
        # Calculate VaR and CVaR at 95% confidence
        var_95 = np.percentile(final_values, 5)
        cvar_95 = final_values[final_values <= var_95].mean()
        
        # Calculate percentile projections
        percentile_data = {}
        time_points = np.arange(0, simulation_paths.shape[1], 21)  # Monthly points
        
        for confidence_level in config.confidence_levels:
            percentile_values = []
            for t in time_points:
                percentile_val = np.percentile(simulation_paths[:, t], confidence_level * 100)
                percentile_values.append(percentile_val)
            
            percentile_data[f"p{int(confidence_level*100)}"] = {
                'time_points': time_points.tolist(),
                'values': percentile_values
            }
        
        # Simulation summary statistics
        simulation_summary = {
            'mean_final_value': float(expected_value),
            'std_final_value': float(np.std(final_values)),
            'min_final_value': float(np.min(final_values)),
            'max_final_value': float(np.max(final_values)),
            'median_final_value': float(np.median(final_values)),
            'skewness': float(stats.skew(final_values)),
            'kurtosis': float(stats.kurtosis(final_values))
        }
        
        return MonteCarloResult(
            config=config,
            expected_value=expected_value,
            probability_of_loss=probability_of_loss,
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            percentile_data=percentile_data,
            simulation_summary=simulation_summary
        )
    
    def _calculate_scenario_loss(self, portfolio: Portfolio, scenario: StressScenario) -> float:
        """Calculate portfolio loss under a stress scenario.
        
        Args:
            portfolio: Portfolio to analyze
            scenario: Stress scenario
            
        Returns:
            Portfolio loss (negative value)
        """
        total_loss = 0.0
        
        for i, ticker in enumerate(portfolio.tickers):
            if ticker in scenario.return_shocks:
                weight = portfolio.weights[i]
                shock = scenario.return_shocks[ticker]
                loss = weight * shock
                total_loss += loss
        
        return total_loss
    
    def _estimate_recovery_time(self, portfolio: Portfolio, loss_magnitude: float) -> Optional[int]:
        """Estimate recovery time from a loss.
        
        Args:
            portfolio: Portfolio to analyze
            loss_magnitude: Magnitude of loss (positive value)
            
        Returns:
            Estimated recovery time in days, or None if cannot estimate
        """
        try:
            # Get historical data to estimate typical returns
            historical_data = self._get_historical_data(portfolio.tickers)
            portfolio_returns = self._calculate_portfolio_returns(historical_data, portfolio.weights)
            
            # Calculate average positive return
            positive_returns = portfolio_returns[portfolio_returns > 0]
            if len(positive_returns) == 0:
                return None
            
            avg_positive_return = positive_returns.mean()
            
            # Estimate recovery time (simplified calculation)
            recovery_days = int(loss_magnitude / avg_positive_return)
            
            # Cap at reasonable maximum (10 years)
            return min(recovery_days, 10 * 252)
            
        except Exception:
            return None
    
    def _calculate_portfolio_returns(self, price_data: pd.DataFrame, weights: List[float]) -> pd.Series:
        """Calculate portfolio returns from price data and weights.
        
        Args:
            price_data: Historical price data
            weights: Portfolio weights
            
        Returns:
            Series of portfolio returns
        """
        returns = price_data.pct_change().dropna()
        portfolio_returns = (returns * weights).sum(axis=1)
        return portfolio_returns
    
    def _calculate_historical_var(self, 
                                portfolio: Portfolio, 
                                confidence_level: float, 
                                time_horizon_days: int) -> VaRResult:
        """Calculate VaR using historical simulation method."""
        # Get historical returns
        historical_data = self._get_historical_data(portfolio.tickers)
        portfolio_returns = self._calculate_portfolio_returns(historical_data, portfolio.weights)
        
        # Scale returns to time horizon
        scaled_returns = portfolio_returns * np.sqrt(time_horizon_days)
        
        # Calculate VaR and CVaR
        var_amount = np.percentile(scaled_returns, confidence_level * 100)
        cvar_amount = scaled_returns[scaled_returns <= var_amount].mean()
        
        return VaRResult(
            portfolio_id=portfolio.id,
            calculation_date=date.today(),
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            var_amount=var_amount,
            cvar_amount=cvar_amount,
            methodology="historical_simulation"
        )
    
    def _calculate_parametric_var(self, 
                                portfolio: Portfolio, 
                                confidence_level: float, 
                                time_horizon_days: int) -> VaRResult:
        """Calculate VaR using parametric (variance-covariance) method."""
        # Get historical data and calculate statistics
        historical_data = self._get_historical_data(portfolio.tickers)
        returns = historical_data.pct_change().dropna()
        
        # Calculate portfolio statistics
        portfolio_returns = self._calculate_portfolio_returns(historical_data, portfolio.weights)
        portfolio_mean = portfolio_returns.mean()
        portfolio_std = portfolio_returns.std()
        
        # Scale to time horizon
        scaled_mean = portfolio_mean * time_horizon_days
        scaled_std = portfolio_std * np.sqrt(time_horizon_days)
        
        # Calculate VaR using normal distribution assumption
        z_score = stats.norm.ppf(confidence_level)
        var_amount = scaled_mean + z_score * scaled_std
        
        # CVaR for normal distribution
        cvar_amount = scaled_mean - scaled_std * stats.norm.pdf(z_score) / confidence_level
        
        return VaRResult(
            portfolio_id=portfolio.id,
            calculation_date=date.today(),
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            var_amount=var_amount,
            cvar_amount=cvar_amount,
            methodology="parametric"
        )
    
    def _calculate_monte_carlo_var(self, 
                                 portfolio: Portfolio, 
                                 confidence_level: float, 
                                 time_horizon_days: int) -> VaRResult:
        """Calculate VaR using Monte Carlo simulation method."""
        # Create Monte Carlo config for VaR calculation
        config = MonteCarloConfig(
            portfolio_tickers=portfolio.tickers,
            portfolio_weights=portfolio.weights,
            time_horizon_years=time_horizon_days / 252.0,
            num_simulations=10000,
            confidence_levels=[confidence_level],
            initial_value=1.0  # Use unit value for return calculation
        )
        
        # Run simulation
        historical_data = self._get_historical_data(portfolio.tickers)
        return_stats = self._calculate_return_statistics(historical_data, portfolio.weights)
        simulation_paths = self._generate_simulation_paths(config, return_stats)
        
        # Calculate returns from simulation paths
        final_values = simulation_paths[:, -1]
        returns = final_values - 1.0  # Convert to returns
        
        # Calculate VaR and CVaR
        var_amount = np.percentile(returns, confidence_level * 100)
        cvar_amount = returns[returns <= var_amount].mean()
        
        return VaRResult(
            portfolio_id=portfolio.id,
            calculation_date=date.today(),
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            var_amount=var_amount,
            cvar_amount=cvar_amount,
            methodology="monte_carlo"
        )
    
    def _calculate_historical_crisis_loss(self, 
                                        portfolio: Portfolio, 
                                        scenario: HistoricalCrisisScenario) -> float:
        """Calculate portfolio loss during a historical crisis period."""
        try:
            # Get historical data for the crisis period
            historical_data = self._get_historical_data(portfolio.tickers)
            
            # Filter data for crisis period
            start_date = pd.to_datetime(scenario.start_date)
            end_date = pd.to_datetime(scenario.end_date)
            
            crisis_data = historical_data.loc[start_date:end_date]
            
            if len(crisis_data) < 2:
                self.logger.warning(f"Insufficient data for crisis period {scenario.name}")
                return 0.0
            
            # Calculate returns during crisis
            crisis_returns = crisis_data.pct_change().dropna()
            portfolio_returns = self._calculate_portfolio_returns(crisis_data, portfolio.weights)
            
            # Calculate cumulative loss
            cumulative_return = (1 + portfolio_returns).prod() - 1
            
            return cumulative_return
            
        except Exception as e:
            self.logger.warning(f"Could not calculate crisis loss for {scenario.name}: {str(e)}")
            return 0.0
    
    def _apply_correlation_shock(self, 
                               correlation_matrix: np.ndarray, 
                               shock_factor: float) -> np.ndarray:
        """Apply correlation shock to correlation matrix.
        
        Args:
            correlation_matrix: Original correlation matrix
            shock_factor: Shock factor (>1.0 increases correlations)
            
        Returns:
            Shocked correlation matrix
        """
        if shock_factor == 1.0:
            return correlation_matrix
        
        # Apply shock while maintaining positive semi-definite property
        shocked_corr = correlation_matrix.copy()
        
        # Increase off-diagonal correlations
        for i in range(len(shocked_corr)):
            for j in range(len(shocked_corr)):
                if i != j:
                    original_corr = shocked_corr[i, j]
                    # Apply shock but cap at reasonable limits
                    shocked_corr[i, j] = np.clip(
                        original_corr * shock_factor, 
                        -0.99, 
                        0.99
                    )
        
        # Ensure matrix remains positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(shocked_corr)
        eigenvals = np.maximum(eigenvals, 0.001)  # Ensure positive eigenvalues
        shocked_corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Normalize diagonal to 1
        np.fill_diagonal(shocked_corr, 1.0)
        
        return shocked_corr