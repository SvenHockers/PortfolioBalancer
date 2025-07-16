# Portfolio Rebalancer

A Python-based automated portfolio rebalancing system that fetches market data, optimizes allocations using Modern Portfolio Theory with age-based glide paths, and executes trades through broker APIs. Built with a modular microservices architecture supporting Docker and Kubernetes deployments.

## Features

- **Automated Data Fetching**: Daily market data retrieval via yfinance API with error handling and backfill capabilities
- **Portfolio Optimization**: Modern Portfolio Theory implementation with Sharpe ratio maximization and age-based glide paths
- **Multi-Broker Support**: Alpaca and Interactive Brokers API integration with paper trading support
- **Flexible Storage**: Parquet file or SQLite database storage options
- **Containerized Deployment**: Docker Compose and Kubernetes support with scheduling
- **Comprehensive Monitoring**: Structured logging, metrics, and health checks
- **Extensible Architecture**: Plugin-based design for custom data providers, optimization strategies, and brokers

## Architecture

The system consists of three core microservices:

1. **Data Fetcher**: Retrieves and stores market data
2. **Portfolio Optimizer**: Calculates optimal allocations using MPT
3. **Trade Executor**: Compares current vs target allocations and executes trades

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- Broker account (Alpaca or Interactive Brokers) for live trading

### Local Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd portfolio-rebalancer
```

2. **Create virtual environment**:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run tests**:
```bash
pytest tests/
```

## Configuration

### Environment Variables

The system is configured through environment variables or a `.env` file. Key configuration sections:

#### Data Configuration
```bash
TICKERS=SPY,QQQ,VTI,VXUS,BND  # Comma-separated ticker list
STORAGE_TYPE=parquet           # "parquet" or "sqlite"
STORAGE_PATH=data             # Data storage directory
BACKFILL_DAYS=252             # Historical data backfill period
```

#### Optimization Configuration
```bash
USER_AGE=35                   # Age for glide path calculation
RISK_FREE_RATE=0.02          # Risk-free rate assumption
LOOKBACK_DAYS=252            # Historical data lookback period
MIN_WEIGHT=0.0               # Minimum asset weight
MAX_WEIGHT=0.4               # Maximum asset weight
SAFE_PORTFOLIO_BONDS=0.8     # Bond allocation in safe portfolio
```

#### Broker Configuration
```bash
# Alpaca (recommended for beginners)
BROKER_TYPE=alpaca
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading

# Interactive Brokers
BROKER_TYPE=ib
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
```

#### Execution Configuration
```bash
REBALANCE_THRESHOLD=0.05     # 5% drift threshold for rebalancing
ORDER_TYPE=market            # "market" or "limit"
EXECUTION_TIME=16:30         # Daily execution time (after market close)
TIMEZONE=America/New_York    # Timezone for scheduling
```

### Configuration File

For complex configurations, create a `config.json` file:

```json
{
  "data": {
    "tickers": ["SPY", "QQQ", "VTI", "VXUS", "BND"],
    "storage_type": "parquet",
    "backfill_days": 252
  },
  "optimization": {
    "user_age": 35,
    "risk_free_rate": 0.02,
    "min_weight": 0.05,
    "max_weight": 0.35
  },
  "executor": {
    "rebalance_threshold": 0.05,
    "order_type": "market"
  }
}
```

## Usage

### Manual Execution

Run individual components:

```bash
# Fetch market data
python -m src.portfolio_rebalancer.fetcher

# Optimize portfolio
python -m src.portfolio_rebalancer.optimizer

# Execute trades (dry run)
python -m src.portfolio_rebalancer.executor --dry-run
```

### Scheduled Execution

The system includes a scheduler for automated daily execution:

```bash
python -m src.portfolio_rebalancer.scheduler
```

## Deployment

### Docker Compose

1. **Create docker-compose.yml**:
```yaml
version: '3.8'

services:
  scheduler:
    build: .
    command: python -m src.portfolio_rebalancer.scheduler
    environment:
      - TICKERS=${TICKERS}
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  data-fetcher:
    build: .
    command: python -m src.portfolio_rebalancer.fetcher
    environment:
      - TICKERS=${TICKERS}
      - STORAGE_PATH=/app/data
    volumes:
      - ./data:/app/data
    profiles: ["manual"]

  optimizer:
    build: .
    command: python -m src.portfolio_rebalancer.optimizer
    environment:
      - USER_AGE=${USER_AGE}
      - RISK_FREE_RATE=${RISK_FREE_RATE}
    volumes:
      - ./data:/app/data
    profiles: ["manual"]

  executor:
    build: .
    command: python -m src.portfolio_rebalancer.executor
    environment:
      - BROKER_TYPE=${BROKER_TYPE}
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
    volumes:
      - ./data:/app/data
    profiles: ["manual"]

volumes:
  data:
  logs:
```

2. **Deploy**:
```bash
# Start scheduler (automatic mode)
docker-compose up -d scheduler

# Run individual services (manual mode)
docker-compose --profile manual up data-fetcher
```

### Kubernetes

1. **Create namespace**:
```bash
kubectl create namespace portfolio-rebalancer
```

2. **Deploy configuration**:
```bash
# Create ConfigMap
kubectl create configmap portfolio-config \
  --from-env-file=.env \
  -n portfolio-rebalancer

# Create Secret for API keys
kubectl create secret generic broker-credentials \
  --from-literal=alpaca-api-key=your_key \
  --from-literal=alpaca-secret-key=your_secret \
  -n portfolio-rebalancer
```

3. **Deploy CronJob**:
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: portfolio-rebalancer
  namespace: portfolio-rebalancer
spec:
  schedule: "30 16 * * 1-5"  # 4:30 PM weekdays
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: rebalancer
            image: portfolio-rebalancer:latest
            command: ["python", "-m", "src.portfolio_rebalancer.scheduler"]
            envFrom:
            - configMapRef:
                name: portfolio-config
            - secretRef:
                name: broker-credentials
            volumeMounts:
            - name: data-storage
              mountPath: /app/data
          volumes:
          - name: data-storage
            persistentVolumeClaim:
              claimName: portfolio-data
          restartPolicy: OnFailure
```

## Customization

### Adding Custom Data Providers

1. **Implement the DataProvider interface**:
```python
from src.portfolio_rebalancer.common.interfaces import DataProvider

class CustomDataProvider(DataProvider):
    def fetch_prices(self, tickers, start_date, end_date):
        # Your custom implementation
        return price_dataframe
```

2. **Register in configuration**:
```python
# In your custom module
from src.portfolio_rebalancer.common.config import get_config

config = get_config()
# Use your custom provider
```

### Adding Custom Optimization Strategies

1. **Implement OptimizationStrategy interface**:
```python
from src.portfolio_rebalancer.common.interfaces import OptimizationStrategy

class CustomOptimizer(OptimizationStrategy):
    def optimize(self, returns, constraints):
        # Your custom optimization logic
        return {"SPY": 0.6, "BND": 0.4}
```

2. **Configure strategy selection**:
```python
# Add strategy selection to config
OPTIMIZATION_STRATEGY=custom
```

### Adding Custom Brokers

1. **Implement BrokerInterface**:
```python
from src.portfolio_rebalancer.common.interfaces import BrokerInterface

class CustomBroker(BrokerInterface):
    def get_positions(self):
        # Fetch current positions
        return {"SPY": 100, "BND": 50}
    
    def place_order(self, symbol, quantity, order_type):
        # Execute trade
        return "order_id_123"
    
    def get_order_status(self, order_id):
        # Check order status
        return "filled"
```

2. **Register broker**:
```bash
BROKER_TYPE=custom
```

### Custom Glide Path Logic

Modify the age-based allocation logic:

```python
class CustomGlidePath:
    def get_allocation_blend(self, age):
        # Custom age-based logic
        if age < 30:
            return (0.9, 0.1)  # 90% aggressive, 10% safe
        elif age < 50:
            return (0.7, 0.3)
        else:
            return (0.5, 0.5)
```

## Monitoring and Observability

### Logging

The system provides structured JSON logging:

```bash
# Set log level
LOG_LEVEL=INFO

# Enable file logging
LOG_FILE_PATH=logs/portfolio_rebalancer.log

# Use text format for development
LOG_FORMAT=text
```

### Metrics

Prometheus metrics are exposed for monitoring:

- `portfolio_rebalancer_execution_time_seconds`
- `portfolio_rebalancer_success_rate`
- `portfolio_rebalancer_portfolio_drift`
- `portfolio_rebalancer_trades_executed_total`

### Health Checks

Health check endpoints for container orchestration:

```bash
# Check service health
curl http://localhost:8080/health

# Check readiness
curl http://localhost:8080/ready
```

## Security Considerations

### API Key Management

- Use environment variables or Kubernetes secrets for API keys
- Enable API key rotation where supported
- Use paper trading URLs for development/testing

### Data Security

- Encrypt sensitive data at rest
- Use secure communication (HTTPS) for all external APIs
- Implement proper access controls in production

### Container Security

- Run containers as non-root user
- Use minimal base images
- Regularly update dependencies

## Troubleshooting

### Common Issues

1. **API Rate Limits**:
   - Reduce fetch frequency
   - Implement exponential backoff
   - Use multiple API keys if available

2. **Optimization Failures**:
   - Check data quality and completeness
   - Adjust optimization constraints
   - Enable fallback to equal-weight allocation

3. **Trade Execution Errors**:
   - Verify broker API credentials
   - Check account permissions and buying power
   - Use paper trading for testing

4. **Data Storage Issues**:
   - Ensure sufficient disk space
   - Check file permissions
   - Validate data integrity

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
LOG_LEVEL=DEBUG
VALIDATE_REQUIRED_CONFIG=true
```

### Testing

Run comprehensive tests:

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs for error details
- Open an issue on GitHub