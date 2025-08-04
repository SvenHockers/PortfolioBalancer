# Portfolio Rebalancer

> **Note: This project is under active development.**
>
> Initial releases may contain bugs, incomplete features, or other issues.  
> We welcome feedback and contributions as the project evolves.

An automated portfolio rebalancing system that fetches market data, optimizes allocations using Modern Portfolio Theory, executes trades through broker APIs, and provides comprehensive analytics and monitoring. Built with Python, containerized with Docker, and deployable on Kubernetes with full observability stack.

## System Architecture

The Portfolio Rebalancer follows a microservices architecture with five core services orchestrated by a central scheduler. The **Scheduler Service** coordinates the daily execution pipeline by triggering the **Data Fetcher** to retrieve market data from Yahoo Finance, the **Portfolio Optimizer** to calculate optimal allocations using Modern Portfolio Theory, the **Trade Executor** to rebalance positions through broker APIs, and the **Analytics Service** to provide comprehensive performance analysis, risk monitoring, and interactive reporting.

Each service operates independently with its own REST API, enabling both automated pipeline execution and manual intervention when needed. The **Analytics Service** provides advanced features including backtesting, Monte Carlo simulations, performance attribution, risk analysis, dividend tracking, and interactive dashboards with real-time monitoring and alerting capabilities.

The system integrates with external APIs including Yahoo Finance for market data and broker APIs for trade execution, while supporting multiple storage backends (Parquet, SQLite, PostgreSQL, Redis) for data persistence. Comprehensive monitoring is built-in through Prometheus metrics collection, Grafana dashboards, Loki log aggregation, and structured JSON logging, making the system production-ready with full observability. The modular design allows for easy customization of data providers, optimization strategies, broker integrations, and analytics engines while maintaining separation of concerns across all pipeline stages.

```mermaid
flowchart TD
    %% External Data Sources
    YF[Yahoo Finance<br/>Market Data API]

    %% Broker APIs
    ALPACA[Alpaca Markets<br/>Trading API]
    IB[Interactive Brokers<br/>Trading API]
    T212[Trading 212<br/>Trading API]

    %% Core Pipeline Services
    SCHEDULER[Scheduler Service<br/>Orchestrates Pipeline<br/>Port 8083]
    FETCHER[Data Fetcher<br/>Market Data Collection<br/>Port 8080]
    OPTIMIZER[Portfolio Optimizer<br/>Modern Portfolio Theory<br/>Port 8081]
    EXECUTOR[Trade Executor<br/>Order Management<br/>Port 8082]
    ANALYTICS[Analytics Service<br/>Performance & Risk Analysis<br/>Port 8084/8085]

    %% Storage Layer
    STORAGE[(Primary Storage<br/>Parquet/SQLite)]
    POSTGRES[(Analytics DB<br/>PostgreSQL)]
    REDIS[(Cache & Queue<br/>Redis)]

    %% Analytics Components
    BACKTESTING[Backtesting<br/>Engine]
    MONTECARLO[Monte Carlo<br/>Simulations]
    RISK[Risk Analysis<br/>& Monitoring]
    PERFORMANCE[Performance<br/>Attribution]
    DIVIDENDS[Dividend<br/>Analysis]

    %% Monitoring & Observability Stack
    PROMETHEUS[Prometheus<br/>Metrics Collection<br/>Port 9090]
    GRAFANA[Grafana<br/>Dashboards & Alerts<br/>Port 3000]
    LOKI[Loki<br/>Log Aggregation<br/>Port 3100]
    PROMTAIL[Promtail<br/>Log Collection]

    %% Async Processing
    CELERY[Celery Workers<br/>Async Analytics]
    WEBSOCKET[WebSocket<br/>Real-time Updates]

    %% Main Pipeline Flow
    SCHEDULER -->|1. Trigger| FETCHER
    SCHEDULER -->|2. Trigger| OPTIMIZER
    SCHEDULER -->|3. Trigger| EXECUTOR
    SCHEDULER -->|4. Trigger| ANALYTICS

    %% Data Flow
    YF -->|Market Data| FETCHER
    FETCHER -->|Store Prices| STORAGE
    STORAGE -->|Historical Data| OPTIMIZER
    STORAGE -->|Historical Data| ANALYTICS
    OPTIMIZER -->|Target Allocation| EXECUTOR

    %% Analytics Flow
    ANALYTICS -->|Results| POSTGRES
    ANALYTICS -->|Cache| REDIS
    ANALYTICS --> BACKTESTING
    ANALYTICS --> MONTECARLO
    ANALYTICS --> RISK
    ANALYTICS --> PERFORMANCE
    ANALYTICS --> DIVIDENDS
    ANALYTICS -->|Tasks| CELERY
    ANALYTICS -->|Real-time| WEBSOCKET

    %% Trading Flow
    EXECUTOR -->|Get Positions| ALPACA
    EXECUTOR -->|Get Positions| IB
    EXECUTOR -->|Get Positions| T212
    EXECUTOR -->|Place Orders| ALPACA
    EXECUTOR -->|Place Orders| IB
    EXECUTOR -->|Place Orders| T212

    %% Monitoring Flow
    FETCHER -.->|Metrics| PROMETHEUS
    OPTIMIZER -.->|Metrics| PROMETHEUS
    EXECUTOR -.->|Metrics| PROMETHEUS
    SCHEDULER -.->|Metrics| PROMETHEUS
    ANALYTICS -.->|Metrics| PROMETHEUS

    PROMETHEUS -->|Visualize| GRAFANA
    GRAFANA -->|Alerts| WEBSOCKET

    FETCHER -.->|Logs| PROMTAIL
    OPTIMIZER -.->|Logs| PROMTAIL
    EXECUTOR -.->|Logs| PROMTAIL
    SCHEDULER -.->|Logs| PROMTAIL
    ANALYTICS -.->|Logs| PROMTAIL

    PROMTAIL -->|Aggregate| LOKI
    LOKI -->|Query| GRAFANA

    %% Styling
    classDef service fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef external fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef storage fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef monitoring fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef analytics fill:#f1f8e9,stroke:#33691e,stroke-width:2px

    class SCHEDULER,FETCHER,OPTIMIZER,EXECUTOR,ANALYTICS service
    class YF,ALPACA,IB,T212 external
    class STORAGE,POSTGRES,REDIS storage
    class PROMETHEUS,GRAFANA,LOKI,PROMTAIL monitoring
    class BACKTESTING,MONTECARLO,RISK,PERFORMANCE,DIVIDENDS,CELERY,WEBSOCKET analytics
```

## Features

### Core Portfolio Management

- **Automated Data Fetching**: Daily market data retrieval with error handling and rate limiting
- **Portfolio Optimization**: Modern Portfolio Theory with Sharpe ratio maximization and age-based glide paths
- **Multi-Broker Support**: Alpaca, Interactive Brokers, and Trading 212 integration with paper trading
- **Flexible Storage**: Parquet files, SQLite, PostgreSQL, or Redis backends
- **Security**: Encrypted credential storage with automatic rotation and secure communication

### Advanced Analytics & Reporting

- **Backtesting Engine**: Historical strategy performance analysis with multiple metrics
- **Monte Carlo Simulations**: Risk assessment and scenario analysis
- **Performance Attribution**: Detailed analysis of returns by asset, sector, and time period
- **Risk Monitoring**: Real-time risk metrics with configurable alerts and thresholds
- **Dividend Analysis**: Comprehensive dividend tracking and yield analysis
- **Interactive Dashboards**: Mobile-responsive Grafana dashboards with zoom, pan, and time selection
- **Automated Reporting**: Scheduled email reports (daily, weekly, monthly, quarterly)

### Production-Ready Infrastructure

- **Containerized Deployment**: Docker Compose and Kubernetes deployment with health checks
- **Comprehensive Monitoring**: Prometheus metrics, Grafana dashboards, and Loki log aggregation
- **Async Processing**: Celery workers for heavy analytics computations
- **Real-time Updates**: WebSocket connections for live dashboard updates
- **High Availability**: Service redundancy, automatic restarts, and circuit breakers
- **Extensible Design**: Plugin architecture for custom providers, strategies, and analytics engines

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Broker account (which is supported in the framework or feel free to integrate new brokers and contribute!)

### Docker Deployment

```bash
# Copy and configure environment
cp examples/config-production.env .env
# Edit .env with your broker credentials and settings

# Start core services (scheduler orchestrates the pipeline)
docker-compose up -d scheduler

# Start with analytics and monitoring (recommended)
docker-compose --profile analytics --profile monitoring up -d

# Or run services manually
docker-compose --profile manual up -d data-fetcher optimizer executor

# Check service health
curl http://localhost:8083/health  # Scheduler
curl http://localhost:8084/health  # Analytics
```

### Kubernetes Deployment

```bash
# Copy and configure environment
cp examples/config-kubernetes.yaml k8s-config.yaml
# Edit yaml file with your broker credentials and settings

# Quick deployment
./scripts/deploy-k8s.sh

# Manual deployment
kubectl create namespace portfolio-rebalancer
kubectl apply -f k8s/

# Update credentials and database passwords
kubectl edit secret broker-credentials -n portfolio-rebalancer
kubectl edit secret analytics-secrets -n portfolio-rebalancer
```

## Configuration

### Essential Environment Variables

```bash
# Data Configuration
TICKERS=SPY,QQQ,VTI,VXUS,BND
STORAGE_TYPE=parquet
BACKFILL_DAYS=252

# Optimization
USER_AGE=35
RISK_FREE_RATE=0.02
OPTIMIZATION_METHOD=sharpe

# Broker Configuration (choose one)
BROKER_TYPE=alpaca  # alpaca, ib, or t212
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Execution
REBALANCE_THRESHOLD=0.05
DRY_RUN=true  # Set to false for live trading
ORDER_TYPE=market

# Scheduling
EXECUTION_TIME=16:30
TIMEZONE=America/New_York

# Analytics Database (PostgreSQL)
ANALYTICS_DB_URL=postgresql://portfolio_user:portfolio_pass@postgres:5432/portfolio_rebalancer
POSTGRES_USER=portfolio_user
POSTGRES_PASSWORD=portfolio_pass
POSTGRES_DB=portfolio_rebalancer

# Redis for Caching and Async Processing
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1

# Security
ENABLE_ENCRYPTION=true
USE_ENCRYPTED_CREDENTIALS=true
ANALYTICS_JWT_SECRET=your-secret-key-change-in-production
```

See `examples/config-development.env` and `examples/config-production.env` for complete configuration examples.

## Analytics Service

The Analytics Service provides comprehensive portfolio analysis, risk monitoring, and interactive reporting capabilities. It operates as a separate microservice with its own database and async processing capabilities.

### Key Components

- **Backtesting Engine**: Historical strategy performance analysis with multiple metrics
- **Monte Carlo Simulator**: Risk assessment and scenario analysis with configurable parameters
- **Performance Tracker**: Real-time portfolio metrics and benchmark comparisons
- **Risk Analyzer**: VaR, Expected Shortfall, and risk decomposition analysis
- **Attribution Engine**: Performance attribution by asset, sector, and time period
- **Dividend Analyzer**: Comprehensive dividend tracking and yield analysis
- **Interactive Reporting**: Mobile-responsive dashboards with real-time updates

### Analytics API Endpoints

```bash
# Performance Analysis
GET /performance/{portfolio_id}           # Current performance metrics
GET /performance/{portfolio_id}/history   # Historical performance data
GET /performance/attribution/{portfolio_id} # Performance attribution

# Risk Analysis
GET /risk/analysis/{portfolio_id}         # Current risk metrics
GET /risk/var/{portfolio_id}              # Value at Risk analysis
GET /risk/stress-test/{portfolio_id}      # Stress test results

# Backtesting
POST /backtest                            # Run backtest with config
GET /backtest/{backtest_id}/results       # Get backtest results
GET /backtest/{backtest_id}/metrics       # Get backtest metrics

# Monte Carlo Simulation
POST /monte-carlo                         # Run Monte Carlo simulation
GET /monte-carlo/{simulation_id}/results  # Get simulation results

# Dividend Analysis
GET /dividends/analysis/{portfolio_id}    # Dividend analysis
GET /dividends/forecast/{portfolio_id}    # Dividend forecasts

# Interactive Reporting
GET /dashboard/config/{portfolio_id}      # Dashboard configuration
POST /alerts/configure                    # Configure alerts
POST /reports/schedule                    # Schedule automated reports
```

### Real-time Features

- **WebSocket Updates**: Live portfolio metrics and alerts via WebSocket on port 8085
- **Async Processing**: Heavy computations handled by Celery workers
- **Caching**: Redis-based caching for improved performance
- **Mobile Support**: Responsive dashboards optimized for mobile devices

## Usage

### Manual Execution

```bash
# Fetch market data
python -m src.portfolio_rebalancer.services.fetcher_service

# Calculate optimal allocations
python -m src.portfolio_rebalancer.services.optimizer_service

# Execute trades (dry run)
python -m src.portfolio_rebalancer.services.executor_service --dry-run

# Run analytics
python -m src.portfolio_rebalancer.services.analytics_service

# Run complete pipeline
python -m src.portfolio_rebalancer.services.scheduler_service --run-once

# Interactive reporting demo
python examples/interactive_reporting_demo.py
```

### REST API

```bash
# Core Services
curl -X POST http://localhost:8080/fetch          # Trigger data fetching
curl http://localhost:8081/optimize               # Get optimization results
curl -X POST http://localhost:8082/execute        # Execute trades
curl http://localhost:8083/status                 # Check pipeline status

# Analytics API
curl http://localhost:8084/health                 # Analytics health check
curl http://localhost:8084/performance/portfolio-1 # Get performance metrics
curl -X POST http://localhost:8084/backtest       # Run backtest
curl http://localhost:8084/risk/analysis          # Get risk analysis
curl http://localhost:8084/dividends/analysis     # Get dividend analysis

# Interactive Reporting
curl http://localhost:8084/dashboard/config       # Get dashboard config
curl -X POST http://localhost:8084/alerts/configure # Configure alerts
curl http://localhost:8084/reports/schedule       # Schedule automated reports
```

### Monitoring & Analytics

```bash
# Check service health
curl http://localhost:8080/health  # Data Fetcher
curl http://localhost:8081/health  # Optimizer
curl http://localhost:8082/health  # Executor
curl http://localhost:8083/health  # Scheduler
curl http://localhost:8084/health  # Analytics

# View Prometheus metrics
curl http://localhost:9090/metrics

# Access Grafana dashboards
open http://localhost:3000  # Default: admin/admin

# Monitor logs with Loki
curl http://localhost:3100/ready

# Monitor service logs
docker-compose logs -f scheduler
docker-compose logs -f analytics
```

## Deployment Profiles

### Development (Basic)

```bash
docker-compose up -d scheduler
# Core services only: scheduler, fetcher, optimizer, executor
# Uses paper trading, debug logging, dry run mode
```

### Analytics-Enabled

```bash
docker-compose --profile analytics up -d
# Includes analytics service, PostgreSQL, Redis, and Celery workers
# Provides backtesting, risk analysis, and performance tracking
```

### Full Production with Monitoring

```bash
docker-compose --profile analytics --profile monitoring up -d
# Complete stack: all services + Prometheus, Grafana, Loki, Promtail
# Access Grafana: http://localhost:3000 (admin/admin)
# Access Prometheus: http://localhost:9090
```

### Manual Services

```bash
docker-compose --profile manual up -d data-fetcher optimizer executor
# Run core services individually without scheduler
```

### Database and Caching

```bash
docker-compose --profile database --profile cache up -d
# PostgreSQL and Redis for advanced storage and caching
```

## Analytics Features

### Performance Analysis

- **Portfolio Metrics**: Total return, Sharpe ratio, alpha, beta, volatility, max drawdown
- **Benchmark Comparison**: Performance vs SPY, sector indices, and custom benchmarks
- **Attribution Analysis**: Returns by asset, sector, time period, and strategy component
- **Risk-Adjusted Returns**: Sortino ratio, Calmar ratio, information ratio

### Risk Management

- **Value at Risk (VaR)**: Historical and parametric VaR calculations
- **Expected Shortfall**: Conditional VaR and tail risk metrics
- **Risk Decomposition**: Contribution to portfolio risk by asset and factor
- **Stress Testing**: Scenario analysis and historical stress tests

### Interactive Dashboards

- **Real-time Monitoring**: Live portfolio metrics with WebSocket updates
- **Mobile-Responsive**: Optimized for tablets and smartphones
- **Interactive Controls**: Zoom, pan, time selection, and crossfilter
- **Automated Alerts**: Email and webhook notifications for risk thresholds

### Backtesting & Simulation

- **Historical Backtesting**: Strategy performance over historical periods
- **Monte Carlo Simulation**: Portfolio outcome distributions and probability analysis
- **Walk-Forward Analysis**: Out-of-sample strategy validation
- **Custom Scenarios**: User-defined market condition testing

## Documentation

- **API Documentation**: [docs/API.md](docs/API.md) - Complete REST API reference
- **Troubleshooting Guide**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Common issues and solutions
- **Monitoring Setup**: [docs/MONITORING.md](docs/MONITORING.md) - Grafana dashboards and alerts
- **Customization Guide**: [docs/CUSTOMIZATION.md](docs/CUSTOMIZATION.md) - Extending the system
- **Configuration Examples**: [examples/](examples/) - Sample configurations for different environments
