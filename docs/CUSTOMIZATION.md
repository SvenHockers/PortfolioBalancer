# Customization Guide

Extend the Portfolio Rebalancer with custom data providers, optimization strategies, brokers, and storage backends.

## Architecture

Modular, plugin-based design with extensible components:

- **Data Providers**: Market data sources (yfinance, Alpha Vantage, custom APIs)
- **Optimization Strategies**: Portfolio algorithms (Sharpe, Risk Parity, Black-Litterman)
- **Broker Interfaces**: Trading platforms (Alpaca, Interactive Brokers, custom)
- **Storage Backends**: Data persistence (Parquet, SQLite, PostgreSQL, Redis, S3)
- **Glide Path Logic**: Age-based allocation strategies
- **Monitoring**: Custom metrics and alerting

## Custom Data Providers

### Basic Implementation
```python
# src/portfolio_rebalancer/fetcher/custom_provider.py
from datetime import date
from typing import List
import pandas as pd
from ..common.interfaces import DataProvider

class CustomDataProvider(DataProvider):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    def fetch_prices(self, tickers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        # Your custom API logic here
        all_data = []
        for ticker in tickers:
            response = self._call_api(ticker, start_date, end_date)
            ticker_data = self._parse_response(response, ticker)
            all_data.append(ticker_data)
        return pd.concat(all_data, ignore_index=False)
```

### Registration
```python
# src/portfolio_rebalancer/fetcher/__init__.py
def get_data_provider(provider_type: str, **kwargs):
    if provider_type == 'custom':
        return CustomDataProvider(**kwargs)
    # ... other providers
```

### Configuration
```bash
DATA_PROVIDER=custom
CUSTOM_API_KEY=your_api_key
CUSTOM_BASE_URL=https://api.example.com/v1
```

## Custom Optimization Strategies

### Basic Implementation
```python
# src/portfolio_rebalancer/optimizer/custom_optimizer.py
from typing import Dict
import pandas as pd
import numpy as np
from ..common.interfaces import OptimizationStrategy

class MomentumOptimizer(OptimizationStrategy):
    def __init__(self, lookback_months: int = 12, top_n: int = 3):
        self.lookback_months = lookback_months
        self.top_n = top_n
    
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        # Calculate momentum scores
        momentum_scores = returns.tail(252).mean() * 252
        top_assets = momentum_scores.nlargest(self.top_n)
        
        # Equal weight among top assets
        weight = 1.0 / len(top_assets)
        return {asset: weight for asset in top_assets.index}

class RiskParityOptimizer(OptimizationStrategy):
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        # Inverse volatility weights
        cov_matrix = returns.cov() * 252
        volatilities = np.sqrt(np.diag(cov_matrix))
        weights = (1 / volatilities) / (1 / volatilities).sum()
        return dict(zip(returns.columns, weights))
```

### Registration
```python
# src/portfolio_rebalancer/optimizer/__init__.py
def get_optimizer(strategy_type: str, **kwargs):
    if strategy_type == 'momentum':
        return MomentumOptimizer(**kwargs)
    elif strategy_type == 'risk_parity':
        return RiskParityOptimizer(**kwargs)
    # ... other optimizers
```

## Custom Broker Interfaces

### Basic Implementation
```python
# src/portfolio_rebalancer/executor/custom_broker.py
from typing import Dict
from ..common.interfaces import BrokerInterface

class TDAmeritradeBroker(BrokerInterface):
    def __init__(self, client_id: str, refresh_token: str):
        self.client_id = client_id
        self.refresh_token = refresh_token
        self._authenticate()
    
    def get_positions(self) -> Dict[str, float]:
        # Fetch positions from TD Ameritrade API
        pass
    
    def place_order(self, symbol: str, quantity: float, order_type: str) -> str:
        # Place order via TD Ameritrade API
        pass
    
    def get_order_status(self, order_id: str) -> str:
        # Check order status
        pass
```

### Configuration
```bash
BROKER_TYPE=td_ameritrade
TD_CLIENT_ID=your_client_id
TD_REFRESH_TOKEN=your_refresh_token
```

## Custom Storage Backends

### Creating a Custom Storage Backend

```python
# src/portfolio_rebalancer/common/custom_storage.py
from typing import List
import pandas as pd
from .interfaces import DataStorage

class RedisStorage(DataStorage):
    """Redis-based storage implementation."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        import redis
        self.redis_client = redis.Redis(host=host, port=port, db=db)
    
    def store_prices(self, data: pd.DataFrame) -> None:
        """Store price data in Redis."""
        for (date, symbol), row in data.iterrows():
            key = f"price:{symbol}:{date}"
            value = {
                'adjusted_close': row['adjusted_close'],
                'volume': row['volume']
            }
            self.redis_client.hset(key, mapping=value)
    
    def get_prices(self, tickers: List[str], lookback_days: int) -> pd.DataFrame:
        """Retrieve price data from Redis."""
        from datetime import date, timedelta
        
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            for ticker in tickers:
                key = f"price:{ticker}:{current_date}"
                data = self.redis_client.hgetall(key)
                
                if data:
                    all_data.append({
                        'date': current_date,
                        'symbol': ticker,
                        'adjusted_close': float(data[b'adjusted_close']),
                        'volume': int(data[b'volume'])
                    })
            
            current_date += timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        return df.set_index(['date', 'symbol'])

class S3Storage(DataStorage):
    """AWS S3-based storage implementation."""
    
    def __init__(self, bucket_name: str, prefix: str = 'portfolio-data/'):
        import boto3
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.prefix = prefix
    
    def store_prices(self, data: pd.DataFrame) -> None:
        """Store price data in S3 as Parquet."""
        import io
        
        # Group by date for efficient storage
        for date_val in data.index.get_level_values('date').unique():
            date_data = data.xs(date_val, level='date')
            
            # Convert to Parquet bytes
            buffer = io.BytesIO()
            date_data.to_parquet(buffer)
            buffer.seek(0)
            
            # Upload to S3
            key = f"{self.prefix}prices/{date_val}.parquet"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue()
            )
    
    def get_prices(self, tickers: List[str], lookback_days: int) -> pd.DataFrame:
        """Retrieve price data from S3."""
        from datetime import date, timedelta
        import io
        
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            key = f"{self.prefix}prices/{current_date}.parquet"
            
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                buffer = io.BytesIO(response['Body'].read())
                date_data = pd.read_parquet(buffer)
                
                # Filter by requested tickers
                date_data = date_data[date_data.index.isin(tickers)]
                
                if not date_data.empty:
                    # Add date level to index
                    date_data['date'] = current_date
                    date_data = date_data.set_index('date', append=True)
                    date_data = date_data.reorder_levels(['date', 'symbol'])
                    all_data.append(date_data)
                    
            except self.s3_client.exceptions.NoSuchKey:
                pass  # Skip missing dates
            
            current_date += timedelta(days=1)
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
```

## Custom Glide Path Logic

### Creating Custom Age-Based Allocation

```python
# src/portfolio_rebalancer/optimizer/custom_glide_path.py
from typing import Tuple

class TargetDateGlidePath:
    """Target date fund style glide path."""
    
    def __init__(self, target_retirement_age: int = 65):
        self.target_retirement_age = target_retirement_age
    
    def get_allocation_blend(self, age: int) -> Tuple[float, float]:
        """Calculate allocation blend based on years to retirement."""
        years_to_retirement = max(0, self.target_retirement_age - age)
        
        if years_to_retirement >= 40:
            # Very young: 90% stocks, 10% bonds
            return (0.9, 0.1)
        elif years_to_retirement >= 30:
            # Young: 80% stocks, 20% bonds
            return (0.8, 0.2)
        elif years_to_retirement >= 20:
            # Middle-aged: 70% stocks, 30% bonds
            return (0.7, 0.3)
        elif years_to_retirement >= 10:
            # Pre-retirement: 60% stocks, 40% bonds
            return (0.6, 0.4)
        elif years_to_retirement >= 5:
            # Near retirement: 50% stocks, 50% bonds
            return (0.5, 0.5)
        else:
            # Retirement: 40% stocks, 60% bonds
            return (0.4, 0.6)

class RiskToleranceGlidePath:
    """Risk tolerance based glide path."""
    
    def __init__(self, risk_tolerance: str = 'moderate'):
        self.risk_tolerance = risk_tolerance.lower()
    
    def get_allocation_blend(self, age: int) -> Tuple[float, float]:
        """Calculate allocation based on age and risk tolerance."""
        base_stock_allocation = max(0.2, (100 - age) / 100)
        
        if self.risk_tolerance == 'aggressive':
            stock_allocation = min(0.95, base_stock_allocation * 1.2)
        elif self.risk_tolerance == 'conservative':
            stock_allocation = base_stock_allocation * 0.8
        else:  # moderate
            stock_allocation = base_stock_allocation
        
        bond_allocation = 1.0 - stock_allocation
        return (stock_allocation, bond_allocation)

class LifecycleGlidePath:
    """Lifecycle-based glide path with multiple phases."""
    
    def get_allocation_blend(self, age: int) -> Tuple[float, float]:
        """Calculate allocation based on life phases."""
        if age < 25:
            # Early career: High growth focus
            return (0.95, 0.05)
        elif age < 35:
            # Career building: Growth with some stability
            return (0.85, 0.15)
        elif age < 45:
            # Peak earning: Balanced growth
            return (0.75, 0.25)
        elif age < 55:
            # Pre-retirement planning: Moderate growth
            return (0.65, 0.35)
        elif age < 65:
            # Near retirement: Capital preservation
            return (0.50, 0.50)
        else:
            # Retirement: Income focus
            return (0.35, 0.65)
```

## Configuration Integration

### Registering Custom Components

```python
# src/portfolio_rebalancer/common/factory.py
from typing import Any, Dict
from .config import get_config

def create_data_provider():
    """Factory function to create data provider based on configuration."""
    config = get_config()
    provider_type = getattr(config.data, 'provider_type', 'yfinance')
    
    if provider_type == 'yfinance':
        from ..fetcher.yfinance_provider import YFinanceProvider
        return YFinanceProvider()
    elif provider_type == 'alpha_vantage':
        from ..fetcher.alpha_vantage_provider import AlphaVantageProvider
        return AlphaVantageProvider(api_key=config.data.alpha_vantage_api_key)
    elif provider_type == 'custom':
        from ..fetcher.custom_provider import CustomDataProvider
        return CustomDataProvider(
            api_key=config.data.custom_api_key,
            base_url=config.data.custom_base_url
        )
    else:
        raise ValueError(f"Unknown data provider: {provider_type}")

def create_optimizer():
    """Factory function to create optimizer based on configuration."""
    config = get_config()
    strategy_type = getattr(config.optimization, 'strategy_type', 'sharpe')
    
    if strategy_type == 'sharpe':
        from ..optimizer.sharpe_optimizer import SharpeOptimizer
        return SharpeOptimizer()
    elif strategy_type == 'momentum':
        from ..optimizer.custom_optimizer import MomentumOptimizer
        return MomentumOptimizer(
            lookback_months=config.optimization.momentum_lookback_months,
            top_n=config.optimization.momentum_top_n
        )
    elif strategy_type == 'risk_parity':
        from ..optimizer.custom_optimizer import RiskParityOptimizer
        return RiskParityOptimizer()
    else:
        raise ValueError(f"Unknown optimization strategy: {strategy_type}")

def create_broker():
    """Factory function to create broker based on configuration."""
    config = get_config()
    broker_type = config.executor.broker_type
    
    if broker_type == 'alpaca':
        from ..executor.alpaca_broker import AlpacaBroker
        return AlpacaBroker(
            api_key=config.broker.alpaca_api_key,
            secret_key=config.broker.alpaca_secret_key,
            base_url=config.broker.alpaca_base_url
        )
    elif broker_type == 'ib':
        from ..executor.ib_broker import IBBroker
        return IBBroker(
            host=config.broker.ib_host,
            port=config.broker.ib_port,
            client_id=config.broker.ib_client_id
        )
    elif broker_type == 'td_ameritrade':
        from ..executor.custom_broker import TDAmeritradeBroker
        return TDAmeritradeBroker(
            client_id=config.broker.td_client_id,
            refresh_token=config.broker.td_refresh_token
        )
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")
```

### Environment Configuration for Custom Components

```bash
# .env file additions for custom components

# Custom Data Provider
DATA_PROVIDER=alpha_vantage
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Custom Optimization Strategy
OPTIMIZATION_STRATEGY=momentum
MOMENTUM_LOOKBACK_MONTHS=12
MOMENTUM_TOP_N=3

# Custom Broker
BROKER_TYPE=td_ameritrade
TD_CLIENT_ID=your_td_client_id
TD_REFRESH_TOKEN=your_td_refresh_token

# Custom Storage
STORAGE_TYPE=redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Custom Glide Path
GLIDE_PATH_TYPE=target_date
TARGET_RETIREMENT_AGE=65
RISK_TOLERANCE=moderate
```

## 🔧 Advanced Customization Examples

### Custom Notification Systems

Create custom notification handlers for alerts and status updates:

```python
# src/portfolio_rebalancer/common/custom_notifications.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class NotificationHandler(ABC):
    """Abstract base class for notification handlers."""
    
    @abstractmethod
    def send_notification(self, message: str, level: str = "info", **kwargs) -> bool:
        """Send notification with specified message and level."""
        pass

class SlackNotificationHandler(NotificationHandler):
    """Slack webhook notification handler."""
    
    def __init__(self, webhook_url: str, channel: str = None):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send_notification(self, message: str, level: str = "info", **kwargs) -> bool:
        """Send notification to Slack channel."""
        color_map = {
            "info": "#36a64f",      # Green
            "warning": "#ff9500",   # Orange
            "error": "#ff0000",     # Red
            "success": "#00ff00"    # Bright Green
        }
        
        payload = {
            "text": f"Portfolio Rebalancer: {message}",
            "attachments": [{
                "color": color_map.get(level, "#36a64f"),
                "fields": [
                    {"title": "Level", "value": level.upper(), "short": True},
                    {"title": "Timestamp", "value": kwargs.get("timestamp", ""), "short": True}
                ]
            }]
        }
        
        if self.channel:
            payload["channel"] = self.channel
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

class EmailNotificationHandler(NotificationHandler):
    """Email notification handler."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, from_email: str, to_emails: list):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    def send_notification(self, message: str, level: str = "info", **kwargs) -> bool:
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            msg['Subject'] = f"Portfolio Rebalancer Alert - {level.upper()}"
            
            body = f"""
            Portfolio Rebalancer Notification
            
            Level: {level.upper()}
            Message: {message}
            Timestamp: {kwargs.get('timestamp', '')}
            
            Additional Details:
            {kwargs.get('details', 'None')}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
        except Exception:
            return False

class DiscordNotificationHandler(NotificationHandler):
    """Discord webhook notification handler."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_notification(self, message: str, level: str = "info", **kwargs) -> bool:
        """Send notification to Discord channel."""
        color_map = {
            "info": 0x3498db,      # Blue
            "warning": 0xf39c12,   # Orange
            "error": 0xe74c3c,     # Red
            "success": 0x2ecc71    # Green
        }
        
        payload = {
            "embeds": [{
                "title": "Portfolio Rebalancer Notification",
                "description": message,
                "color": color_map.get(level, 0x3498db),
                "fields": [
                    {"name": "Level", "value": level.upper(), "inline": True},
                    {"name": "Timestamp", "value": kwargs.get("timestamp", ""), "inline": True}
                ]
            }]
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 204
        except Exception:
            return False
```

### Custom Metrics and Monitoring

Extend the monitoring system with custom metrics:

```python
# src/portfolio_rebalancer/common/custom_metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Dict, Any
import time
from functools import wraps

# Custom metrics
portfolio_performance_gauge = Gauge(
    'portfolio_performance_total_return', 
    'Total portfolio return percentage',
    ['strategy', 'time_period']
)

trade_execution_counter = Counter(
    'trades_executed_total',
    'Total number of trades executed',
    ['symbol', 'action', 'broker']
)

optimization_duration_histogram = Histogram(
    'optimization_duration_seconds',
    'Time spent on portfolio optimization',
    ['strategy', 'num_assets']
)

data_quality_gauge = Gauge(
    'data_quality_score',
    'Data quality score (0-1)',
    ['provider', 'symbol']
)

rebalance_threshold_gauge = Gauge(
    'portfolio_drift_percentage',
    'Current portfolio drift from target allocation',
    ['symbol']
)

class CustomMetricsCollector:
    """Custom metrics collection and reporting."""
    
    def __init__(self):
        self.start_times = {}
    
    def record_portfolio_performance(self, strategy: str, time_period: str, return_pct: float):
        """Record portfolio performance metrics."""
        portfolio_performance_gauge.labels(
            strategy=strategy, 
            time_period=time_period
        ).set(return_pct)
    
    def record_trade_execution(self, symbol: str, action: str, broker: str):
        """Record trade execution metrics."""
        trade_execution_counter.labels(
            symbol=symbol,
            action=action,
            broker=broker
        ).inc()
    
    def record_data_quality(self, provider: str, symbol: str, quality_score: float):
        """Record data quality metrics."""
        data_quality_gauge.labels(
            provider=provider,
            symbol=symbol
        ).set(quality_score)
    
    def record_portfolio_drift(self, symbol: str, drift_pct: float):
        """Record portfolio drift metrics."""
        rebalance_threshold_gauge.labels(symbol=symbol).set(drift_pct)
    
    def time_optimization(self, strategy: str, num_assets: int):
        """Context manager for timing optimization operations."""
        return OptimizationTimer(strategy, num_assets)

class OptimizationTimer:
    """Context manager for timing optimization operations."""
    
    def __init__(self, strategy: str, num_assets: int):
        self.strategy = strategy
        self.num_assets = num_assets
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        optimization_duration_histogram.labels(
            strategy=self.strategy,
            num_assets=str(self.num_assets)
        ).observe(duration)

# Decorator for automatic metrics collection
def collect_metrics(metric_type: str):
    """Decorator to automatically collect metrics for functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                # Record success metrics
                if metric_type == 'optimization':
                    duration = time.time() - start_time
                    optimization_duration_histogram.labels(
                        strategy=kwargs.get('strategy', 'unknown'),
                        num_assets=str(kwargs.get('num_assets', 0))
                    ).observe(duration)
                return result
            except Exception as e:
                # Record error metrics
                error_counter = Counter(
                    f'{metric_type}_errors_total',
                    f'Total {metric_type} errors',
                    ['error_type']
                )
                error_counter.labels(error_type=type(e).__name__).inc()
                raise
        return wrapper
    return decorator
```

### Custom Risk Management

Implement advanced risk management features:

```python
# src/portfolio_rebalancer/risk/custom_risk_management.py
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import date, timedelta

class RiskManager:
    """Advanced risk management system."""
    
    def __init__(self, max_portfolio_var: float = 0.05, max_individual_weight: float = 0.4):
        self.max_portfolio_var = max_portfolio_var
        self.max_individual_weight = max_individual_weight
    
    def assess_portfolio_risk(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
        """Assess overall portfolio risk metrics."""
        # Convert weights to numpy array
        weight_array = np.array([weights.get(col, 0) for col in returns.columns])
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weight_array).sum(axis=1)
        
        # Risk metrics
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
        portfolio_var = np.percentile(portfolio_returns, 5)  # 5% VaR
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        
        return {
            'portfolio_volatility': portfolio_vol,
            'value_at_risk_5pct': portfolio_var,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'concentration_risk': max(weights.values()) if weights else 0
        }
    
    def check_risk_limits(self, risk_metrics: Dict[str, float]) -> List[str]:
        """Check if portfolio violates risk limits."""
        violations = []
        
        if risk_metrics['value_at_risk_5pct'] < -self.max_portfolio_var:
            violations.append(f"Portfolio VaR ({risk_metrics['value_at_risk_5pct']:.3f}) exceeds limit ({-self.max_portfolio_var:.3f})")
        
        if risk_metrics['concentration_risk'] > self.max_individual_weight:
            violations.append(f"Concentration risk ({risk_metrics['concentration_risk']:.3f}) exceeds limit ({self.max_individual_weight:.3f})")
        
        if risk_metrics['max_drawdown'] < -0.2:  # 20% max drawdown
            violations.append(f"Max drawdown ({risk_metrics['max_drawdown']:.3f}) exceeds 20% limit")
        
        return violations
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def suggest_risk_adjustments(self, weights: Dict[str, float], violations: List[str]) -> Dict[str, float]:
        """Suggest weight adjustments to address risk violations."""
        adjusted_weights = weights.copy()
        
        # If concentration risk is too high, cap individual weights
        max_weight = max(weights.values()) if weights else 0
        if max_weight > self.max_individual_weight:
            # Find the asset with highest weight
            max_asset = max(weights, key=weights.get)
            excess_weight = max_weight - self.max_individual_weight
            
            # Reduce the overweight asset
            adjusted_weights[max_asset] = self.max_individual_weight
            
            # Distribute excess weight to other assets
            other_assets = [k for k in weights.keys() if k != max_asset]
            if other_assets:
                weight_per_asset = excess_weight / len(other_assets)
                for asset in other_assets:
                    adjusted_weights[asset] += weight_per_asset
        
        # Normalize weights to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights

class VolatilityTargeting:
    """Volatility targeting risk management."""
    
    def __init__(self, target_volatility: float = 0.12):
        self.target_volatility = target_volatility
    
    def adjust_weights_for_volatility(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust portfolio weights to target specific volatility."""
        # Calculate current portfolio volatility
        weight_array = np.array([weights.get(col, 0) for col in returns.columns])
        portfolio_returns = (returns * weight_array).sum(axis=1)
        current_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Calculate scaling factor
        if current_vol > 0:
            scale_factor = self.target_volatility / current_vol
            # Cap scaling to reasonable bounds
            scale_factor = min(max(scale_factor, 0.5), 2.0)
        else:
            scale_factor = 1.0
        
        # Apply scaling (this is simplified - in practice you'd adjust individual weights)
        adjusted_weights = {k: v * scale_factor for k, v in weights.items()}
        
        # Normalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
```

### Custom Configuration Validation

Create advanced configuration validation:

```python
# src/portfolio_rebalancer/common/custom_validation.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, validator, Field
import re
from datetime import datetime

class AdvancedPortfolioConfig(BaseModel):
    """Advanced portfolio configuration with custom validation."""
    
    tickers: List[str] = Field(..., min_items=1, max_items=50)
    target_allocations: Optional[Dict[str, float]] = None
    rebalance_schedule: str = Field("daily", regex=r"^(daily|weekly|monthly|quarterly)$")
    risk_budget: float = Field(0.15, ge=0.05, le=0.30)
    max_turnover: float = Field(0.50, ge=0.10, le=1.00)
    
    @validator('tickers')
    def validate_tickers(cls, v):
        """Validate ticker symbols."""
        ticker_pattern = re.compile(r'^[A-Z]{1,5}$')
        invalid_tickers = [ticker for ticker in v if not ticker_pattern.match(ticker)]
        if invalid_tickers:
            raise ValueError(f"Invalid ticker symbols: {invalid_tickers}")
        return v
    
    @validator('target_allocations')
    def validate_allocations(cls, v, values):
        """Validate target allocations sum to 1 and match tickers."""
        if v is None:
            return v
        
        # Check if allocations sum to approximately 1
        total_allocation = sum(v.values())
        if not (0.99 <= total_allocation <= 1.01):
            raise ValueError(f"Target allocations sum to {total_allocation}, must sum to 1.0")
        
        # Check if all tickers have allocations
        tickers = values.get('tickers', [])
        missing_tickers = set(tickers) - set(v.keys())
        if missing_tickers:
            raise ValueError(f"Missing allocations for tickers: {missing_tickers}")
        
        extra_tickers = set(v.keys()) - set(tickers)
        if extra_tickers:
            raise ValueError(f"Allocations for unknown tickers: {extra_tickers}")
        
        return v
    
    @validator('risk_budget')
    def validate_risk_budget(cls, v):
        """Validate risk budget is reasonable."""
        if v > 0.25:
            raise ValueError("Risk budget above 25% is considered too aggressive")
        return v

class TradingHoursValidator:
    """Validate trading hours and market schedules."""
    
    @staticmethod
    def validate_execution_time(execution_time: str, timezone: str) -> bool:
        """Validate execution time is during market hours."""
        # This is a simplified example - in practice you'd check actual market hours
        try:
            hour, minute = map(int, execution_time.split(':'))
            # US market hours: 9:30 AM - 4:00 PM ET
            if timezone == "America/New_York":
                market_open = 9 * 60 + 30  # 9:30 AM in minutes
                market_close = 16 * 60     # 4:00 PM in minutes
                execution_minutes = hour * 60 + minute
                
                # Allow execution after market close for end-of-day rebalancing
                return execution_minutes >= market_close
            return True
        except ValueError:
            return False

class BrokerConfigValidator:
    """Validate broker-specific configurations."""
    
    @staticmethod
    def validate_alpaca_config(api_key: str, secret_key: str, base_url: str) -> List[str]:
        """Validate Alpaca configuration."""
        errors = []
        
        if not api_key or len(api_key) < 20:
            errors.append("Alpaca API key appears to be invalid")
        
        if not secret_key or len(secret_key) < 40:
            errors.append("Alpaca secret key appears to be invalid")
        
        valid_urls = [
            "https://api.alpaca.markets",
            "https://paper-api.alpaca.markets"
        ]
        if base_url not in valid_urls:
            errors.append(f"Invalid Alpaca base URL. Must be one of: {valid_urls}")
        
        return errors
    
    @staticmethod
    def validate_ib_config(host: str, port: int, client_id: int) -> List[str]:
        """Validate Interactive Brokers configuration."""
        errors = []
        
        if not re.match(r'^(\d{1,3}\.){3}\d{1,3}$', host) and host != 'localhost':
            errors.append("Invalid IB host format")
        
        valid_ports = [7496, 7497, 4001, 4002]  # Common IB ports
        if port not in valid_ports:
            errors.append(f"Unusual IB port {port}. Common ports are: {valid_ports}")
        
        if not (0 <= client_id <= 32):
            errors.append("IB client ID should be between 0 and 32")
        
        return errors
```

This comprehensive customization guide provides developers with extensive examples and patterns for extending the Portfolio Rebalancer system while maintaining code quality, security, and performance standards.