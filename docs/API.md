# API Reference

REST APIs for all Portfolio Rebalancer services with consistent response formats and error handling.

## Service Endpoints

| Service | Port | Base URL |
|---------|------|----------|
| Data Fetcher | 8080 | `http://localhost:8080` |
| Portfolio Optimizer | 8081 | `http://localhost:8081` |
| Trade Executor | 8082 | `http://localhost:8082` |
| Scheduler | 8083 | `http://localhost:8083` |

## Response Format

```json
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully",
  "timestamp": "2025-01-16T10:30:00Z"
}
```

## Core Endpoints

### Health Checks
- **GET** `/health` - Service health status
- **GET** `/ready` - Readiness for container orchestration

### Data Fetcher (Port 8080)
- **POST** `/fetch` - Trigger market data collection
- **GET** `/data/{ticker}` - Get historical price data
- **GET** `/quality` - Data quality metrics

### Portfolio Optimizer (Port 8081)
- **POST** `/optimize` - Calculate optimal allocation
- **GET** `/allocation` - Current target allocation
- **GET** `/risk` - Risk analysis and metrics

### Trade Executor (Port 8082)
- **POST** `/execute` - Execute rebalancing trades
- **GET** `/positions` - Current portfolio positions
- **GET** `/orders/{order_id}` - Order status
- **GET** `/drift` - Portfolio drift analysis

### Scheduler (Port 8083)
- **GET** `/status` - Pipeline execution status
- **POST** `/execute` - Trigger manual execution
- **GET** `/history` - Execution history
- **PUT** `/schedule` - Update execution schedule

## Quick Examples

### Trigger Portfolio Rebalancing
```bash
curl -X POST http://localhost:8083/execute \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'
```

### Get Current Positions
```bash
curl http://localhost:8082/positions
```

### Optimize Portfolio
```bash
curl -X POST http://localhost:8081/optimize \
  -H "Content-Type: application/json" \
  -d '{"user_age": 35, "optimization_method": "sharpe"}'
```

## Error Codes

| Code | Description | Status |
|------|-------------|--------|
| `VALIDATION_ERROR` | Request validation failed | 400 |
| `AUTHENTICATION_ERROR` | Authentication failed | 401 |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded | 429 |
| `BROKER_ERROR` | Broker API error | 502 |
| `OPTIMIZATION_FAILED` | Portfolio optimization failed | 500 |

## Rate Limiting

- **Default**: 100 requests/minute per IP
- **Burst**: 20 requests/second
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`