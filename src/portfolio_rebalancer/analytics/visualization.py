"""Visualization data generation for Monte Carlo results."""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .models import MonteCarloResult
from .exceptions import VisualizationError


@dataclass
class VisualizationConfig:
    """Configuration for visualization data generation."""
    
    chart_type: str
    time_resolution: str = "monthly"
    include_percentiles: List[float] = None
    num_points: int = 100
    include_confidence_bands: bool = True
    
    def __post_init__(self):
        if self.include_percentiles is None:
            self.include_percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]


class VisualizationDataGenerator:
    """Generator for various visualization data formats from Monte Carlo results."""
    
    def __init__(self):
        """Initialize visualization data generator."""
        self.supported_chart_types = [
            'fan_chart', 'probability_cone', 'histogram', 'scatter',
            'return_distribution', 'drawdown_analysis', 'risk_return_scatter'
        ]
    
    def generate_fan_chart_data(self, 
                              result: MonteCarloResult, 
                              config: VisualizationConfig) -> Dict[str, Any]:
        """
        Generate fan chart data showing percentile bands over time.
        
        Args:
            result: Monte Carlo simulation result
            config: Visualization configuration
            
        Returns:
            Fan chart data structure
        """
        try:
            # Extract time series data
            time_points, time_labels = self._generate_time_series(result, config)
            
            # Generate percentile bands
            percentile_bands = self._generate_percentile_bands(result, config, time_points)
            
            # Calculate expected path
            expected_path = self._calculate_expected_path(result, time_points)
            
            return {
                "chart_type": "fan_chart",
                "metadata": {
                    "title": "Portfolio Value Projection Fan Chart",
                    "subtitle": f"{result.config.time_horizon_years}-year Monte Carlo simulation",
                    "num_simulations": result.config.num_simulations,
                    "initial_value": result.config.initial_value
                },
                "time_axis": {
                    "labels": time_labels,
                    "points": time_points,
                    "resolution": config.time_resolution
                },
                "data": {
                    "percentile_bands": percentile_bands,
                    "expected_path": expected_path,
                    "median_path": self._extract_percentile_path(result, 0.5, time_points),
                    "confidence_intervals": self._generate_confidence_intervals(percentile_bands)
                },
                "styling": {
                    "colors": self._get_fan_chart_colors(len(percentile_bands)),
                    "opacity_gradient": True,
                    "show_grid": True
                }
            }
            
        except Exception as e:
            raise VisualizationError(f"Failed to generate fan chart data: {str(e)}")
    
    def generate_probability_cone_data(self, 
                                     result: MonteCarloResult, 
                                     config: VisualizationConfig) -> Dict[str, Any]:
        """
        Generate probability cone data showing confidence intervals.
        
        Args:
            result: Monte Carlo simulation result
            config: Visualization configuration
            
        Returns:
            Probability cone data structure
        """
        try:
            time_points, time_labels = self._generate_time_series(result, config)
            
            # Generate symmetric confidence intervals
            confidence_intervals = []
            
            # Create confidence intervals from percentiles
            sorted_percentiles = sorted(config.include_percentiles)
            
            for i, lower_p in enumerate(sorted_percentiles):
                if lower_p >= 0.5:
                    break
                    
                upper_p = 1.0 - lower_p
                if upper_p in sorted_percentiles:
                    confidence_level = upper_p - lower_p
                    
                    lower_path = self._extract_percentile_path(result, lower_p, time_points)
                    upper_path = self._extract_percentile_path(result, upper_p, time_points)
                    
                    confidence_intervals.append({
                        "confidence_level": confidence_level,
                        "label": f"{int(confidence_level * 100)}% Confidence",
                        "lower_bound": lower_path,
                        "upper_bound": upper_path,
                        "area_color": self._get_confidence_color(confidence_level),
                        "border_style": "solid" if confidence_level >= 0.8 else "dashed"
                    })
            
            # Sort by confidence level (widest first for proper layering)
            confidence_intervals.sort(key=lambda x: x["confidence_level"], reverse=True)
            
            return {
                "chart_type": "probability_cone",
                "metadata": {
                    "title": "Portfolio Value Probability Cone",
                    "subtitle": f"Confidence intervals from {result.config.num_simulations:,} simulations",
                    "time_horizon_years": result.config.time_horizon_years
                },
                "time_axis": {
                    "labels": time_labels,
                    "points": time_points,
                    "resolution": config.time_resolution
                },
                "data": {
                    "confidence_intervals": confidence_intervals,
                    "median_projection": self._extract_percentile_path(result, 0.5, time_points),
                    "expected_projection": self._calculate_expected_path(result, time_points),
                    "initial_value": result.config.initial_value
                },
                "risk_metrics": {
                    "probability_of_loss": result.probability_of_loss,
                    "value_at_risk_95": result.value_at_risk_95,
                    "expected_shortfall": result.conditional_var_95
                }
            }
            
        except Exception as e:
            raise VisualizationError(f"Failed to generate probability cone data: {str(e)}")
    
    def generate_return_distribution_data(self, 
                                        result: MonteCarloResult, 
                                        config: VisualizationConfig) -> Dict[str, Any]:
        """
        Generate return distribution histogram data.
        
        Args:
            result: Monte Carlo simulation result
            config: Visualization configuration
            
        Returns:
            Return distribution data structure
        """
        try:
            # Calculate returns from final values
            initial_value = result.config.initial_value
            time_horizon = result.config.time_horizon_years
            
            # Generate synthetic return distribution from summary statistics
            simulation_summary = result.simulation_summary or {}
            mean_final = simulation_summary.get('mean_final_value', result.expected_value)
            std_final = simulation_summary.get('std_final_value', mean_final * 0.2)
            
            # Convert to annualized returns
            mean_return = ((mean_final / initial_value) ** (1 / time_horizon)) - 1
            return_volatility = (std_final / initial_value) / (time_horizon ** 0.5)
            
            # Generate histogram bins
            num_bins = min(50, max(20, result.config.num_simulations // 200))
            
            # Estimate return range
            min_return = mean_return - 4 * return_volatility
            max_return = mean_return + 4 * return_volatility
            
            bin_edges = np.linspace(min_return, max_return, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Generate approximate histogram (would use actual simulation data in practice)
            bin_counts = self._generate_normal_histogram(
                bin_centers, mean_return, return_volatility, result.config.num_simulations
            )
            
            # Calculate key statistics
            percentile_returns = {
                f"p{int(p*100)}": mean_return + return_volatility * self._normal_ppf(p)
                for p in [0.05, 0.25, 0.5, 0.75, 0.95]
            }
            
            return {
                "chart_type": "return_distribution",
                "metadata": {
                    "title": "Annualized Return Distribution",
                    "subtitle": f"Distribution of {time_horizon}-year annualized returns",
                    "num_simulations": result.config.num_simulations
                },
                "histogram": {
                    "bin_edges": bin_edges.tolist(),
                    "bin_centers": bin_centers.tolist(),
                    "bin_counts": bin_counts.tolist(),
                    "bin_width": bin_edges[1] - bin_edges[0]
                },
                "statistics": {
                    "mean_return": mean_return,
                    "median_return": percentile_returns["p50"],
                    "return_volatility": return_volatility,
                    "skewness": simulation_summary.get('skewness', 0.0),
                    "kurtosis": simulation_summary.get('kurtosis', 3.0),
                    "percentiles": percentile_returns
                },
                "risk_metrics": {
                    "probability_of_negative_return": self._calculate_negative_return_probability(
                        mean_return, return_volatility
                    ),
                    "downside_deviation": return_volatility * 0.7,  # Approximation
                    "value_at_risk_5pct": percentile_returns["p5"]
                },
                "styling": {
                    "bar_color": "#3498db",
                    "mean_line_color": "#e74c3c",
                    "percentile_colors": {
                        "p5": "#e74c3c",
                        "p95": "#27ae60"
                    }
                }
            }
            
        except Exception as e:
            raise VisualizationError(f"Failed to generate return distribution data: {str(e)}")
    
    def generate_drawdown_analysis_data(self, 
                                      result: MonteCarloResult, 
                                      config: VisualizationConfig) -> Dict[str, Any]:
        """
        Generate drawdown analysis visualization data.
        
        Args:
            result: Monte Carlo simulation result
            config: Visualization configuration
            
        Returns:
            Drawdown analysis data structure
        """
        try:
            time_points, time_labels = self._generate_time_series(result, config)
            
            # Generate drawdown paths from percentile data
            percentile_paths = {}
            for p in config.include_percentiles:
                percentile_paths[f"p{int(p*100)}"] = self._extract_percentile_path(result, p, time_points)
            
            # Calculate drawdowns for each percentile path
            drawdown_data = {}
            for percentile, path in percentile_paths.items():
                drawdowns = self._calculate_drawdown_series(path)
                drawdown_data[percentile] = {
                    "drawdown_series": drawdowns,
                    "max_drawdown": min(drawdowns),
                    "recovery_periods": self._calculate_recovery_periods(drawdowns)
                }
            
            # Calculate aggregate drawdown statistics
            max_drawdowns = [data["max_drawdown"] for data in drawdown_data.values()]
            
            return {
                "chart_type": "drawdown_analysis",
                "metadata": {
                    "title": "Portfolio Drawdown Analysis",
                    "subtitle": "Maximum drawdown scenarios across percentiles",
                    "time_horizon_years": result.config.time_horizon_years
                },
                "time_axis": {
                    "labels": time_labels,
                    "points": time_points,
                    "resolution": config.time_resolution
                },
                "data": {
                    "drawdown_paths": drawdown_data,
                    "underwater_curves": self._generate_underwater_curves(drawdown_data),
                    "recovery_analysis": self._analyze_recovery_patterns(drawdown_data)
                },
                "statistics": {
                    "worst_case_drawdown": min(max_drawdowns),
                    "median_max_drawdown": np.median(max_drawdowns),
                    "average_recovery_time": self._calculate_average_recovery_time(drawdown_data),
                    "drawdown_frequency": self._calculate_drawdown_frequency(drawdown_data)
                }
            }
            
        except Exception as e:
            raise VisualizationError(f"Failed to generate drawdown analysis data: {str(e)}")
    
    def _generate_time_series(self, 
                            result: MonteCarloResult, 
                            config: VisualizationConfig) -> Tuple[List[int], List[str]]:
        """Generate time points and labels for visualization."""
        
        total_days = result.config.time_horizon_years * 252
        
        if config.time_resolution == "daily":
            step = max(1, total_days // config.num_points)
        elif config.time_resolution == "weekly":
            step = 5
        elif config.time_resolution == "monthly":
            step = 21
        elif config.time_resolution == "quarterly":
            step = 63
        else:
            step = 21  # Default to monthly
        
        time_points = list(range(0, total_days + 1, step))
        if time_points[-1] != total_days:
            time_points.append(total_days)
        
        # Generate date labels
        start_date = date.today()
        time_labels = []
        for days in time_points:
            future_date = start_date + timedelta(days=days)
            time_labels.append(future_date.isoformat())
        
        return time_points, time_labels
    
    def _generate_percentile_bands(self, 
                                 result: MonteCarloResult, 
                                 config: VisualizationConfig, 
                                 time_points: List[int]) -> List[Dict[str, Any]]:
        """Generate percentile bands for fan chart."""
        
        bands = []
        
        for percentile in sorted(config.include_percentiles):
            path = self._extract_percentile_path(result, percentile, time_points)
            
            bands.append({
                "percentile": percentile,
                "label": f"{int(percentile * 100)}th percentile",
                "values": path,
                "color": self._get_percentile_color(percentile),
                "opacity": self._get_percentile_opacity(percentile)
            })
        
        return bands
    
    def _extract_percentile_path(self, 
                               result: MonteCarloResult, 
                               percentile: float, 
                               time_points: List[int]) -> List[float]:
        """Extract percentile path from Monte Carlo result."""
        
        percentile_key = f"p{int(percentile * 100)}"
        percentile_data = result.percentile_data or {}
        
        if percentile_key in percentile_data:
            stored_values = percentile_data[percentile_key].get("values", [])
            stored_time_points = percentile_data[percentile_key].get("time_points", [])
            
            # Interpolate to match requested time points
            if len(stored_values) > 0 and len(stored_time_points) > 0:
                return self._interpolate_path(stored_time_points, stored_values, time_points)
        
        # Fallback: generate synthetic path
        return self._generate_synthetic_path(result, percentile, time_points)
    
    def _interpolate_path(self, 
                        source_times: List[int], 
                        source_values: List[float], 
                        target_times: List[int]) -> List[float]:
        """Interpolate path values to target time points."""
        
        if len(source_times) != len(source_values):
            raise ValueError("Source times and values must have same length")
        
        # Use numpy for interpolation
        interpolated = np.interp(target_times, source_times, source_values)
        return interpolated.tolist()
    
    def _generate_synthetic_path(self, 
                               result: MonteCarloResult, 
                               percentile: float, 
                               time_points: List[int]) -> List[float]:
        """Generate synthetic path when actual data is not available."""
        
        initial_value = result.config.initial_value
        time_horizon_years = result.config.time_horizon_years
        
        # Estimate growth parameters
        expected_final = result.expected_value
        annual_growth = (expected_final / initial_value) ** (1 / time_horizon_years) - 1
        
        # Adjust for percentile
        percentile_adjustment = self._normal_ppf(percentile) * 0.1  # 10% volatility assumption
        adjusted_growth = annual_growth + percentile_adjustment
        
        # Generate path
        path = []
        for days in time_points:
            years = days / 252.0
            value = initial_value * ((1 + adjusted_growth) ** years)
            path.append(value)
        
        return path
    
    def _calculate_expected_path(self, 
                               result: MonteCarloResult, 
                               time_points: List[int]) -> List[float]:
        """Calculate expected value path over time."""
        
        initial_value = result.config.initial_value
        final_expected = result.expected_value
        time_horizon_years = result.config.time_horizon_years
        
        # Calculate compound annual growth rate
        cagr = (final_expected / initial_value) ** (1 / time_horizon_years) - 1
        
        # Generate expected path
        expected_path = []
        for days in time_points:
            years = days / 252.0
            expected_value = initial_value * ((1 + cagr) ** years)
            expected_path.append(expected_value)
        
        return expected_path
    
    def _generate_confidence_intervals(self, percentile_bands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate confidence intervals from percentile bands."""
        
        intervals = []
        sorted_bands = sorted(percentile_bands, key=lambda x: x["percentile"])
        
        # Create symmetric intervals
        for i, lower_band in enumerate(sorted_bands):
            if lower_band["percentile"] >= 0.5:
                break
            
            # Find corresponding upper band
            upper_percentile = 1.0 - lower_band["percentile"]
            upper_band = next(
                (b for b in sorted_bands if abs(b["percentile"] - upper_percentile) < 0.01),
                None
            )
            
            if upper_band:
                confidence_level = upper_percentile - lower_band["percentile"]
                intervals.append({
                    "confidence_level": confidence_level,
                    "lower_values": lower_band["values"],
                    "upper_values": upper_band["values"],
                    "fill_color": self._get_confidence_color(confidence_level)
                })
        
        return intervals
    
    def _calculate_drawdown_series(self, value_path: List[float]) -> List[float]:
        """Calculate drawdown series from value path."""
        
        if not value_path:
            return []
        
        drawdowns = []
        running_max = value_path[0]
        
        for value in value_path:
            running_max = max(running_max, value)
            drawdown = (value - running_max) / running_max if running_max > 0 else 0
            drawdowns.append(drawdown)
        
        return drawdowns
    
    def _calculate_recovery_periods(self, drawdown_series: List[float]) -> List[int]:
        """Calculate recovery periods from drawdown series."""
        
        recovery_periods = []
        in_drawdown = False
        drawdown_start = 0
        
        for i, drawdown in enumerate(drawdown_series):
            if drawdown < -0.01 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                drawdown_start = i
            elif drawdown >= -0.01 and in_drawdown:  # End of drawdown
                recovery_period = i - drawdown_start
                recovery_periods.append(recovery_period)
                in_drawdown = False
        
        return recovery_periods
    
    def _generate_normal_histogram(self, 
                                 bin_centers: np.ndarray, 
                                 mean: float, 
                                 std: float, 
                                 total_count: int) -> List[int]:
        """Generate histogram counts assuming normal distribution."""
        
        # Calculate probability density for each bin
        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
        
        probabilities = []
        for center in bin_centers:
            # Normal probability density
            prob = np.exp(-0.5 * ((center - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            probabilities.append(prob * bin_width)
        
        # Normalize and scale to total count
        total_prob = sum(probabilities)
        if total_prob > 0:
            counts = [int(p * total_count / total_prob) for p in probabilities]
        else:
            counts = [0] * len(bin_centers)
        
        return counts
    
    def _normal_ppf(self, percentile: float) -> float:
        """Approximate normal distribution percent point function."""
        # Simple approximation for normal PPF
        if percentile <= 0:
            return -np.inf
        elif percentile >= 1:
            return np.inf
        elif percentile == 0.5:
            return 0.0
        else:
            # Beasley-Springer-Moro approximation
            if percentile < 0.5:
                sign = -1
                r = percentile
            else:
                sign = 1
                r = 1 - percentile
            
            t = np.sqrt(-2 * np.log(r))
            result = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
            
            return sign * result
    
    def _calculate_negative_return_probability(self, mean_return: float, return_volatility: float) -> float:
        """Calculate probability of negative returns."""
        if return_volatility <= 0:
            return 1.0 if mean_return < 0 else 0.0
        
        z_score = -mean_return / return_volatility
        # Approximate normal CDF
        return 0.5 * (1 + np.tanh(z_score * np.sqrt(2 / np.pi)))
    
    def _get_fan_chart_colors(self, num_bands: int) -> List[str]:
        """Get color palette for fan chart bands."""
        # Generate color palette from blue to red
        colors = []
        for i in range(num_bands):
            ratio = i / max(1, num_bands - 1)
            # Interpolate between blue and red
            r = int(52 + ratio * (231 - 52))    # 52 -> 231
            g = int(152 + ratio * (76 - 152))   # 152 -> 76  
            b = int(219 + ratio * (60 - 219))   # 219 -> 60
            colors.append(f"rgb({r},{g},{b})")
        
        return colors
    
    def _get_percentile_color(self, percentile: float) -> str:
        """Get color for specific percentile."""
        if percentile <= 0.05:
            return "#e74c3c"  # Red
        elif percentile <= 0.25:
            return "#f39c12"  # Orange
        elif percentile <= 0.75:
            return "#3498db"  # Blue
        elif percentile <= 0.95:
            return "#2ecc71"  # Green
        else:
            return "#27ae60"  # Dark green
    
    def _get_percentile_opacity(self, percentile: float) -> float:
        """Get opacity for specific percentile."""
        # More extreme percentiles get higher opacity
        distance_from_median = abs(percentile - 0.5)
        return 0.3 + 0.7 * (distance_from_median * 2)
    
    def _get_confidence_color(self, confidence_level: float) -> str:
        """Get color for confidence interval."""
        if confidence_level >= 0.9:
            return "rgba(52, 152, 219, 0.2)"  # Light blue
        elif confidence_level >= 0.8:
            return "rgba(46, 204, 113, 0.2)"  # Light green
        elif confidence_level >= 0.5:
            return "rgba(241, 196, 15, 0.2)"  # Light yellow
        else:
            return "rgba(231, 76, 60, 0.2)"   # Light red
    
    def _generate_underwater_curves(self, drawdown_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Generate underwater curves showing time spent in drawdown."""
        
        underwater_curves = {}
        
        for percentile, data in drawdown_data.items():
            drawdown_series = data["drawdown_series"]
            underwater_curve = []
            
            for drawdown in drawdown_series:
                # Underwater curve shows cumulative time in drawdown
                if drawdown < -0.01:  # In drawdown
                    underwater_curve.append(1.0)
                else:
                    underwater_curve.append(0.0)
            
            underwater_curves[percentile] = underwater_curve
        
        return underwater_curves
    
    def _analyze_recovery_patterns(self, drawdown_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze recovery patterns across percentiles."""
        
        all_recovery_periods = []
        percentile_recoveries = {}
        
        for percentile, data in drawdown_data.items():
            recovery_periods = data["recovery_periods"]
            all_recovery_periods.extend(recovery_periods)
            percentile_recoveries[percentile] = {
                "avg_recovery": np.mean(recovery_periods) if recovery_periods else 0,
                "max_recovery": max(recovery_periods) if recovery_periods else 0,
                "num_drawdowns": len(recovery_periods)
            }
        
        return {
            "overall_avg_recovery": np.mean(all_recovery_periods) if all_recovery_periods else 0,
            "overall_max_recovery": max(all_recovery_periods) if all_recovery_periods else 0,
            "percentile_analysis": percentile_recoveries,
            "total_drawdown_events": len(all_recovery_periods)
        }
    
    def _calculate_average_recovery_time(self, drawdown_data: Dict[str, Any]) -> float:
        """Calculate average recovery time across all scenarios."""
        
        all_recoveries = []
        for data in drawdown_data.values():
            all_recoveries.extend(data["recovery_periods"])
        
        return np.mean(all_recoveries) if all_recoveries else 0.0
    
    def _calculate_drawdown_frequency(self, drawdown_data: Dict[str, Any]) -> float:
        """Calculate frequency of drawdown events."""
        
        total_events = sum(len(data["recovery_periods"]) for data in drawdown_data.values())
        total_scenarios = len(drawdown_data)
        
        return total_events / total_scenarios if total_scenarios > 0 else 0.0