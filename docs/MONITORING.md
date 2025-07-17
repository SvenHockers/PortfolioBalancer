# Monitoring Guide

Built-in observability with Prometheus metrics, structured logging, and Grafana dashboards.

## Features

- **Prometheus Metrics**: Performance and business metrics at `/metrics`
- **Structured Logging**: JSON logs with correlation IDs
- **Health Endpoints**: `/health` and `/ready` for container orchestration
- **Grafana Dashboards**: Pre-configured templates
- **Alert Rules**: Automated alerting for critical issues

## Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `portfolio_rebalancer_execution_time_seconds` | Histogram | Pipeline execution time |
| `portfolio_rebalancer_success_rate` | Gauge | Success rate of executions |
| `portfolio_rebalancer_portfolio_drift` | Gauge | Portfolio drift from target |
| `portfolio_rebalancer_trades_executed_total` | Counter | Total trades executed |
| `portfolio_rebalancer_portfolio_value` | Gauge | Total portfolio value |
| `portfolio_rebalancer_sharpe_ratio` | Gauge | Current Sharpe ratio |

## Configuration

```bash
# Enable monitoring
ENABLE_METRICS=true
METRICS_PORT=8000
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Quick Setup

### Docker Compose
```bash
# Start monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Kubernetes
```yaml
# Liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8083
  initialDelaySeconds: 30
  periodSeconds: 30
```

## Alert Rules

### Critical Alerts
- Service down for >2 minutes
- Portfolio loss >$10,000
- Optimization convergence failure

### Warning Alerts
- High execution time (>5 minutes)
- Success rate <90%
- Portfolio drift >10%
- High broker API error rate

### Configuration
```yaml
# monitoring/alert_rules.yml
- alert: PortfolioServiceDown
  expr: up{job="portfolio-rebalancer"} == 0
  for: 2m
  labels:
    severity: critical

- alert: HighPortfolioDrift
  expr: portfolio_rebalancer_portfolio_drift > 0.1
  for: 15m
  labels:
    severity: warning
```

## Access Points

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093