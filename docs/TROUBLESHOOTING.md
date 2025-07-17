# Troubleshooting Guide

Quick solutions for common Portfolio Rebalancer issues.

## Quick Health Check

```bash
# System diagnostics
python --version  # Must be 3.9+
python -c "from src.portfolio_rebalancer.common.config import get_config; print('✅ Config OK')"

# Service health
for port in 8080 8081 8082 8083; do
  curl -s http://localhost:$port/health && echo "✅ Port $port OK" || echo "❌ Port $port down"
done
```

## Common Issues

### Configuration Issues

**Problem**: Service fails to start, missing environment variables
```bash
# Quick fix
cp .env.example .env
# Edit .env with your values

# Validate config
python -c "from src.portfolio_rebalancer.common.config import get_config; get_config()"
```

**Problem**: Invalid ticker symbols
```bash
# Test tickers
python -c "
import yfinance as yf
for ticker in ['SPY', 'QQQ', 'VTI']:
    data = yf.download(ticker, period='5d', progress=False)
    print(f'{ticker}: {"✅" if not data.empty else "❌"}')
"
```

### Data Fetching Issues

**Problem**: API rate limits, network timeouts
```bash
# Test connectivity
curl -I "https://query1.finance.yahoo.com/v8/finance/chart/SPY"

# Reduce load
TICKERS=SPY,QQQ  # Fewer tickers
BACKFILL_DAYS=30  # Less history
FETCH_DELAY_SECONDS=1  # Rate limiting
```

### Optimization Issues

**Problem**: Optimization fails to converge
```bash
# Check data quality
python -c "
import pandas as pd
data = pd.read_parquet('data/prices.parquet')
returns = data.pct_change().dropna()
print(f'Data shape: {data.shape}')
print(f'NaN values: {returns.isna().sum().sum()}')
"

# Solutions
COVARIANCE_REGULARIZATION=0.01  # Increase regularization
LOOKBACK_DAYS=126  # Reduce lookback
OPTIMIZATION_METHOD=equal_weight  # Fallback method
```

### Broker Connection Issues

**Problem**: Alpaca authentication failed
```bash
# Test credentials
python -c "
import os
from alpaca.trading.client import TradingClient
client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
print(f'✅ Account: {client.get_account().status}')
"

# Fix URLs
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading
```

**Problem**: Interactive Brokers connection refused
```bash
# Check if TWS/Gateway is running
netstat -an | grep 7497

# Configure
IB_HOST=127.0.0.1
IB_PORT=7497  # Paper trading port
```

### Docker Issues

**Problem**: Container startup failures, permission errors
```bash
# Check logs
docker-compose logs scheduler

# Fix permissions
mkdir -p data logs
sudo chown -R 1000:1000 data logs
chmod -R 755 data logs

# Increase resources
# In docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 1G
```

### Performance Issues

**Problem**: High memory usage, slow optimization
```bash
# Monitor usage
docker stats

# Optimize
LOOKBACK_DAYS=126  # Reduce data
TICKERS=SPY,QQQ,VTI,BND  # Fewer assets
MAX_MEMORY_USAGE_MB=1024  # Set limits
```

## Debug Mode

```bash
# Enable debugging
DEBUG=true
LOG_LEVEL=DEBUG
LOG_FORMAT=text

# Collect logs
docker-compose logs > debug.log 2>&1
grep -i "error\|failed\|exception" debug.log
```

## Getting Help

Include when reporting issues:
- System info (OS, Python version)
- Sanitized configuration
- Error messages and logs
- Steps to reproduce

Never share API keys or sensitive data.