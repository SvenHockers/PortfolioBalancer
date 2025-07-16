# Customization Guide

This guide explains how to extend and customize the Portfolio Rebalancer system to meet specific requirements.

## Architecture Overview

The system is built with a plugin-based architecture that allows easy customization of:
- Data providers (market data sources)
- Optimization strategies (portfolio allocation algorithms)
- Broker interfaces (trading execution)
- Storage backends (data persistence)
- Glide path logic (age-based allocation)

## Custom Data Providers

### Creating a Custom Data Provider

1. **Implement the DataProvider interface**:

```python
# src/portfolio_rebalancer/fetcher/custom_provider.py
from datetime import date
from typing import List
import pandas as pd
from ..common.interfaces import DataProvider

class CustomDataProvider(DataProvider):
    """Custom data provider implementation."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    def fetch_prices(self, tickers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch price data from custom API."""
        all_data = []
        
        for ticker in tickers:
            # Your custom API logic here
            response = self._call_api(ticker, start_date, end_date)
            ticker_data = self._parse_response(response, ticker)
            all_data.append(ticker_data)
        
        return pd.concat(all_data, ignore_index=False)
    
    def _call_api(self, ticker: str, start_date: date, end_date: date):
        """Make API call to custom data source."""
        import requests
        
        url = f"{self.base_url}/prices/{ticker}"
        params = {
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'api_key': self.api_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def _parse_response(self, response: dict, ticker: str) -> pd.DataFrame:
        """Parse API response into expected DataFrame format."""
        data = []
        for item in response['prices']:
            data.append({
                'date': pd.to_datetime(item['date']).date(),
                'symbol': ticker,
                'adjusted_close': float(item['close']),
                'volume': int(item['volume'])
            })
        
        df = pd.DataFrame(data)
        return df.set_index(['date', 'symbol'])
```

2. **Register the custom provider**:

```python
# src/portfolio_rebalancer/fetcher/__init__.py
from .yfinance_provider import YFinanceProvider
from .custom_provider import CustomDataProvider

def get_data_provider(provider_type: str, **kwargs):
    """Factory function for data providers."""
    if provider_type == 'yfinance':
        return YFinanceProvider(**kwargs)
    elif provider_type == 'custom':
        return CustomDataProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
```

3. **Configure the provider**:

```bash
# In .env file
DATA_PROVIDER=custom
CUSTOM_API_KEY=your_api_key
CUSTOM_BASE_URL=https://api.example.com/v1
```

### Example: Alpha Vantage Provider

```python
# src/portfolio_rebalancer/fetcher/alpha_vantage_provider.py
import time
from datetime import date
from typing import List
import pandas as pd
import requests
from ..common.interfaces import DataProvider

class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider implementation."""
    
    def __init__(self, api_key: str, calls_per_minute: int = 5):
        self.api_key = api_key
        self.calls_per_minute = calls_per_minute
        self.call_interval = 60 / calls_per_minute
        self.last_call_time = 0
    
    def fetch_prices(self, tickers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch price data from Alpha Vantage API."""
        all_data = []
        
        for ticker in tickers:
            self._rate_limit()
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'outputsize': 'full',
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                ticker_data = self._parse_alpha_vantage_data(data, ticker, start_date, end_date)
                all_data.append(ticker_data)
        
        return pd.concat(all_data, ignore_index=False) if all_data else pd.DataFrame()
    
    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.call_interval:
            time.sleep(self.call_interval - time_since_last_call)
        
        self.last_call_time = time.time()
    
    def _parse_alpha_vantage_data(self, data: dict, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Parse Alpha Vantage response data."""
        time_series = data['Time Series (Daily)']
        parsed_data = []
        
        for date_str, values in time_series.items():
            price_date = pd.to_datetime(date_str).date()
            
            if start_date <= price_date <= end_date:
                parsed_data.append({
                    'date': price_date,
                    'symbol': ticker,
                    'adjusted_close': float(values['5. adjusted close']),
                    'volume': int(values['6. volume'])
                })
        
        df = pd.DataFrame(parsed_data)
        return df.set_index(['date', 'symbol'])
```

## Custom Optimization Strategies

### Creating a Custom Optimizer

1. **Implement the OptimizationStrategy interface**:

```python
# src/portfolio_rebalancer/optimizer/custom_optimizer.py
from typing import Dict
import pandas as pd
import numpy as np
from ..common.interfaces import OptimizationStrategy

class MomentumOptimizer(OptimizationStrategy):
    """Momentum-based optimization strategy."""
    
    def __init__(self, lookback_months: int = 12, top_n: int = 3):
        self.lookback_months = lookback_months
        self.top_n = top_n
    
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """Optimize portfolio based on momentum signals."""
        # Calculate momentum scores (12-month returns)
        momentum_scores = returns.tail(252).mean() * 252  # Annualized returns
        
        # Select top N assets by momentum
        top_assets = momentum_scores.nlargest(self.top_n)
        
        # Equal weight among top assets
        weight = 1.0 / len(top_assets)
        allocations = {asset: weight for asset in top_assets.index}
        
        # Fill remaining assets with zero weight
        for asset in returns.columns:
            if asset not in allocations:
                allocations[asset] = 0.0
        
        return allocations

class RiskParityOptimizer(OptimizationStrategy):
    """Risk parity optimization strategy."""
    
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """Optimize portfolio using risk parity approach."""
        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252  # Annualized
        
        # Calculate inverse volatility weights
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol_weights = 1 / volatilities
        
        # Normalize to sum to 1
        weights = inv_vol_weights / inv_vol_weights.sum()
        
        # Apply constraints
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()  # Renormalize
        
        return dict(zip(returns.columns, weights))

class BlackLittermanOptimizer(OptimizationStrategy):
    """Black-Litterman optimization with views."""
    
    def __init__(self, tau: float = 0.025, risk_aversion: float = 3.0):
        self.tau = tau
        self.risk_aversion = risk_aversion
    
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """Optimize using Black-Litterman model."""
        # Market cap weights (simplified - equal weights as prior)
        n_assets = len(returns.columns)
        w_market = np.ones(n_assets) / n_assets
        
        # Calculate expected returns and covariance
        mu = returns.mean() * 252
        sigma = returns.cov() * 252
        
        # Implied equilibrium returns
        pi = self.risk_aversion * sigma @ w_market
        
        # Black-Litterman formula (without views for simplicity)
        tau_sigma = self.tau * sigma
        bl_mu = np.linalg.inv(tau_sigma + np.linalg.inv(sigma)) @ (tau_sigma @ mu + np.linalg.inv(sigma) @ pi)
        bl_sigma = np.linalg.inv(np.linalg.inv(tau_sigma) + np.linalg.inv(sigma))
        
        # Optimize portfolio
        inv_sigma = np.linalg.inv(bl_sigma)
        ones = np.ones((n_assets, 1))
        
        # Calculate optimal weights
        w_opt = (inv_sigma @ bl_mu) / (ones.T @ inv_sigma @ bl_mu)
        w_opt = w_opt.flatten()
        
        # Apply constraints
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        
        w_opt = np.clip(w_opt, min_weight, max_weight)
        w_opt = w_opt / w_opt.sum()
        
        return dict(zip(returns.columns, w_opt))
```

2. **Register the custom optimizer**:

```python
# src/portfolio_rebalancer/optimizer/__init__.py
from .sharpe_optimizer import SharpeOptimizer
from .custom_optimizer import MomentumOptimizer, RiskParityOptimizer, BlackLittermanOptimizer

def get_optimizer(strategy_type: str, **kwargs):
    """Factory function for optimization strategies."""
    if strategy_type == 'sharpe':
        return SharpeOptimizer(**kwargs)
    elif strategy_type == 'momentum':
        return MomentumOptimizer(**kwargs)
    elif strategy_type == 'risk_parity':
        return RiskParityOptimizer(**kwargs)
    elif strategy_type == 'black_litterman':
        return BlackLittermanOptimizer(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
```

## Custom Broker Interfaces

### Creating a Custom Broker

1. **Implement the BrokerInterface**:

```python
# src/portfolio_rebalancer/executor/custom_broker.py
from typing import Dict
from ..common.interfaces import BrokerInterface

class TDAmeritradeBroker(BrokerInterface):
    """TD Ameritrade broker implementation."""
    
    def __init__(self, client_id: str, refresh_token: str):
        self.client_id = client_id
        self.refresh_token = refresh_token
        self.access_token = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with TD Ameritrade API."""
        import requests
        
        url = "https://api.tdameritrade.com/v1/oauth2/token"
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.client_id
        }
        
        response = requests.post(url, data=data)
        response.raise_for_status()
        
        self.access_token = response.json()['access_token']
    
    def get_positions(self) -> Dict[str, float]:
        """Get current portfolio positions."""
        import requests
        
        url = "https://api.tdameritrade.com/v1/accounts"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        accounts = response.json()
        positions = {}
        
        for account in accounts:
            for position in account.get('positions', []):
                symbol = position['instrument']['symbol']
                quantity = float(position['longQuantity']) - float(position['shortQuantity'])
                positions[symbol] = quantity
        
        return positions
    
    def place_order(self, symbol: str, quantity: float, order_type: str) -> str:
        """Place a trade order."""
        import requests
        import uuid
        
        account_id = self._get_account_id()
        url = f"https://api.tdameritrade.com/v1/accounts/{account_id}/orders"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        order_data = {
            'orderType': order_type.upper(),
            'session': 'NORMAL',
            'duration': 'DAY',
            'orderStrategyType': 'SINGLE',
            'orderLegCollection': [{
                'instruction': 'BUY' if quantity > 0 else 'SELL',
                'quantity': abs(quantity),
                'instrument': {
                    'symbol': symbol,
                    'assetType': 'EQUITY'
                }
            }]
        }
        
        response = requests.post(url, headers=headers, json=order_data)
        response.raise_for_status()
        
        # Extract order ID from response headers
        order_id = response.headers.get('Location', '').split('/')[-1]
        return order_id or str(uuid.uuid4())
    
    def get_order_status(self, order_id: str) -> str:
        """Get status of a placed order."""
        import requests
        
        account_id = self._get_account_id()
        url = f"https://api.tdameritrade.com/v1/accounts/{account_id}/orders/{order_id}"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        order_data = response.json()
        return order_data.get('status', 'UNKNOWN').lower()
    
    def _get_account_id(self) -> str:
        """Get the primary account ID."""
        import requests
        
        url = "https://api.tdameritrade.com/v1/accounts"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        accounts = response.json()
        return accounts[0]['securitiesAccount']['accountId']
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

This customization guide provides comprehensive examples for extending the Portfolio Rebalancer system with custom components while maintaining the existing architecture and interfaces.