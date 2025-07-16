"""Age-based glide path implementation for portfolio allocation."""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GlidePathConfig:
    """Configuration for age-based glide path calculations."""
    
    # Age thresholds for allocation changes
    young_age_threshold: int = 30
    middle_age_threshold: int = 50
    retirement_age: int = 65
    
    # Aggressive portfolio allocation (stocks/equity)
    young_aggressive_allocation: float = 0.9
    middle_aggressive_allocation: float = 0.7
    retirement_aggressive_allocation: float = 0.4
    
    # Safe portfolio composition (bonds/fixed income)
    safe_portfolio_bonds: float = 0.8
    safe_portfolio_cash: float = 0.2
    
    # Transition smoothing
    use_smooth_transitions: bool = True


class GlidePath:
    """
    Age-based asset allocation glide path implementation.
    
    Implements a configurable glide path that blends between aggressive
    (equity-heavy) and safe (bond-heavy) portfolios based on investor age.
    """
    
    def __init__(self, config: Optional[GlidePathConfig] = None):
        """
        Initialize glide path with configuration.
        
        Args:
            config: GlidePathConfig object, uses defaults if None
        """
        self.config = config or GlidePathConfig()
        logger.info(f"GlidePath initialized with config: {self.config}")
    
    def get_allocation_blend(self, age: int) -> Tuple[float, float]:
        """
        Calculate allocation blend between aggressive and safe portfolios.
        
        Args:
            age: Investor age in years
            
        Returns:
            Tuple of (aggressive_weight, safe_weight) that sum to 1.0
        """
        if age < 0:
            raise ValueError("Age cannot be negative")
        
        if age > 120:
            logger.warning(f"Age {age} seems unusually high, proceeding with calculation")
        
        # Calculate aggressive allocation percentage based on age
        aggressive_percentage = self._calculate_aggressive_percentage(age)
        
        # Convert to blend weights
        aggressive_weight = aggressive_percentage
        safe_weight = 1.0 - aggressive_percentage
        
        logger.debug(f"Age {age}: {aggressive_weight:.2%} aggressive, {safe_weight:.2%} safe")
        
        return aggressive_weight, safe_weight
    
    def _calculate_aggressive_percentage(self, age: int) -> float:
        """
        Calculate the percentage allocation to aggressive portfolio based on age.
        
        Args:
            age: Investor age in years
            
        Returns:
            Percentage allocation to aggressive portfolio (0.0 to 1.0)
        """
        config = self.config
        
        if age <= config.young_age_threshold:
            # Young investors: high aggressive allocation
            return config.young_aggressive_allocation
        
        elif age <= config.middle_age_threshold:
            # Transition from young to middle age
            if config.use_smooth_transitions:
                return self._linear_interpolation(
                    age,
                    config.young_age_threshold, config.middle_age_threshold,
                    config.young_aggressive_allocation, config.middle_aggressive_allocation
                )
            else:
                return config.middle_aggressive_allocation
        
        elif age <= config.retirement_age:
            # Transition from middle age to retirement
            if config.use_smooth_transitions:
                return self._linear_interpolation(
                    age,
                    config.middle_age_threshold, config.retirement_age,
                    config.middle_aggressive_allocation, config.retirement_aggressive_allocation
                )
            else:
                return config.retirement_aggressive_allocation
        
        else:
            # Post-retirement: conservative allocation
            return config.retirement_aggressive_allocation
    
    def _linear_interpolation(self, 
                            age: int,
                            age_start: int, 
                            age_end: int,
                            allocation_start: float, 
                            allocation_end: float) -> float:
        """
        Perform linear interpolation between two age points.
        
        Args:
            age: Current age
            age_start: Starting age for interpolation
            age_end: Ending age for interpolation
            allocation_start: Allocation at starting age
            allocation_end: Allocation at ending age
            
        Returns:
            Interpolated allocation percentage
        """
        if age_end == age_start:
            return allocation_start
        
        # Linear interpolation formula
        progress = (age - age_start) / (age_end - age_start)
        interpolated = allocation_start + progress * (allocation_end - allocation_start)
        
        # Ensure result is within valid bounds
        return max(0.0, min(1.0, interpolated))
    
    def blend_portfolios(self, 
                        aggressive_allocation: Dict[str, float],
                        safe_allocation: Dict[str, float],
                        age: int) -> Dict[str, float]:
        """
        Blend aggressive and safe portfolio allocations based on age.
        
        Args:
            aggressive_allocation: Dictionary of aggressive portfolio weights
            safe_allocation: Dictionary of safe portfolio weights
            age: Investor age in years
            
        Returns:
            Dictionary of blended portfolio weights
        """
        # Get blend weights
        aggressive_weight, safe_weight = self.get_allocation_blend(age)
        
        # Get all unique tickers from both portfolios
        all_tickers = set(aggressive_allocation.keys()) | set(safe_allocation.keys())
        
        # Blend the allocations
        blended_allocation = {}
        
        for ticker in all_tickers:
            aggressive_weight_for_ticker = aggressive_allocation.get(ticker, 0.0)
            safe_weight_for_ticker = safe_allocation.get(ticker, 0.0)
            
            blended_weight = (
                aggressive_weight * aggressive_weight_for_ticker +
                safe_weight * safe_weight_for_ticker
            )
            
            if blended_weight > 0:
                blended_allocation[ticker] = blended_weight
        
        # Normalize to ensure weights sum to 1.0
        total_weight = sum(blended_allocation.values())
        if total_weight > 0:
            blended_allocation = {
                ticker: weight / total_weight 
                for ticker, weight in blended_allocation.items()
            }
        
        logger.info(f"Blended portfolio for age {age}: {len(blended_allocation)} assets")
        
        return blended_allocation
    
    def get_safe_portfolio_allocation(self, bond_tickers: list, cash_tickers: list = None) -> Dict[str, float]:
        """
        Generate safe portfolio allocation based on configuration.
        
        Args:
            bond_tickers: List of bond/fixed income ticker symbols
            cash_tickers: List of cash equivalent ticker symbols (optional)
            
        Returns:
            Dictionary of safe portfolio weights
        """
        if not bond_tickers:
            raise ValueError("Bond tickers cannot be empty for safe portfolio")
        
        cash_tickers = cash_tickers or []
        
        safe_allocation = {}
        
        # Allocate to bonds
        if bond_tickers:
            bond_weight_per_ticker = self.config.safe_portfolio_bonds / len(bond_tickers)
            for ticker in bond_tickers:
                safe_allocation[ticker] = bond_weight_per_ticker
        
        # Allocate to cash equivalents
        if cash_tickers:
            cash_weight_per_ticker = self.config.safe_portfolio_cash / len(cash_tickers)
            for ticker in cash_tickers:
                safe_allocation[ticker] = cash_weight_per_ticker
        
        # If no cash tickers provided, allocate all to bonds
        if not cash_tickers and bond_tickers:
            bond_weight_per_ticker = 1.0 / len(bond_tickers)
            safe_allocation = {ticker: bond_weight_per_ticker for ticker in bond_tickers}
        
        logger.debug(f"Safe portfolio allocation: {safe_allocation}")
        
        return safe_allocation
    
    def validate_allocation(self, allocation: Dict[str, float], tolerance: float = 1e-6) -> bool:
        """
        Validate that allocation weights sum to approximately 1.0.
        
        Args:
            allocation: Dictionary of asset weights
            tolerance: Tolerance for weight sum validation
            
        Returns:
            True if allocation is valid
        """
        if not allocation:
            return False
        
        total_weight = sum(allocation.values())
        
        if abs(total_weight - 1.0) > tolerance:
            logger.error(f"Allocation weights sum to {total_weight:.6f}, expected 1.0")
            return False
        
        # Check for negative weights
        negative_weights = [ticker for ticker, weight in allocation.items() if weight < 0]
        if negative_weights:
            logger.error(f"Negative weights found for tickers: {negative_weights}")
            return False
        
        return True
    
    def get_age_based_recommendation(self, age: int) -> str:
        """
        Get human-readable recommendation based on age.
        
        Args:
            age: Investor age in years
            
        Returns:
            String description of recommended allocation strategy
        """
        aggressive_weight, safe_weight = self.get_allocation_blend(age)
        
        if age <= self.config.young_age_threshold:
            return (f"Young investor (age {age}): Aggressive growth strategy with "
                   f"{aggressive_weight:.0%} in growth assets, {safe_weight:.0%} in safe assets")
        
        elif age <= self.config.middle_age_threshold:
            return (f"Middle-aged investor (age {age}): Balanced growth strategy with "
                   f"{aggressive_weight:.0%} in growth assets, {safe_weight:.0%} in safe assets")
        
        elif age <= self.config.retirement_age:
            return (f"Pre-retirement investor (age {age}): Conservative strategy with "
                   f"{aggressive_weight:.0%} in growth assets, {safe_weight:.0%} in safe assets")
        
        else:
            return (f"Retirement-age investor (age {age}): Capital preservation strategy with "
                   f"{aggressive_weight:.0%} in growth assets, {safe_weight:.0%} in safe assets")