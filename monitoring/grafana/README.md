# Portfolio Analytics Grafana Dashboards

This directory contains comprehensive Grafana dashboards for portfolio analytics, providing rich visualizations and monitoring capabilities for portfolio performance, risk analysis, backtesting results, and dividend income tracking.

## Dashboard Overview

### 1. Portfolio Performance Dashboard (`portfolio-performance-comprehensive.json`)
- **Purpose**: Comprehensive portfolio performance monitoring with benchmark comparisons
- **Key Features**:
  - Performance overview with key metrics (total return, Sharpe ratio, volatility, max drawdown)
  - Cumulative returns vs multiple benchmarks (S&P 500, Total Market, NASDAQ 100)
  - Rolling performance metrics (30-day Sharpe, volatility, alpha)
  - Drawdown analysis and excess returns tracking
  - Performance attribution analysis (allocation, selection, interaction effects)
  - Benchmark comparison table with color-coded metrics
  - Portfolio allocation drift monitoring
  - Performance alerts and notifications

### 2. Risk Analytics Dashboard (`risk-analytics-dashboard.json`)
- **Purpose**: Comprehensive risk monitoring with VaR metrics and correlation analysis
- **Key Features**:
  - Risk overview with VaR, CVaR, beta, tracking error, and concentration risk
  - Value at Risk (VaR) analysis at 95% and 99% confidence levels
  - Risk factor exposure analysis (market, size, value, momentum factors)
  - Asset correlation heatmap with color-coded correlation matrix
  - Sector concentration risk visualization
  - Rolling risk metrics (volatility, beta, tracking error)
  - Tail risk analysis (maximum drawdown, recovery time, pain index)
  - Geographic exposure breakdown
  - Risk limit monitoring with breach alerts
  - Risk alerts and notifications

### 3. Backtesting Results Dashboard (`backtesting-results-dashboard.json`)
- **Purpose**: Strategy comparison and backtesting analysis
- **Key Features**:
  - Strategy performance overview with key metrics comparison
  - Strategy comparison with cumulative returns vs benchmark
  - Risk-return scatter plot for strategy visualization
  - Drawdown comparison across strategies
  - Rolling Sharpe ratio comparison
  - Strategy performance metrics table with color-coded results
  - Transaction cost analysis and turnover rates
  - Strategy allocation evolution over time
  - Walk-forward analysis results heatmap
  - Statistical significance tests for strategy comparison
  - Backtest execution logs

### 4. Dividend Income Dashboard (`dividend-income-dashboard.json`)
- **Purpose**: Income-focused analytics with yield tracking and projections
- **Key Features**:
  - Income overview with current yield, projected income, growth rates
  - Dividend income history (monthly, quarterly, annual)
  - Yield tracking over time vs benchmarks
  - Top dividend contributors analysis
  - Dividend yield by holding with sustainability metrics
  - Income projections (conservative, moderate, optimistic scenarios)
  - Dividend coverage analysis with risk assessment
  - Seasonal income patterns
  - Income goal tracking and progress monitoring
  - Dividend growth rates by holding
  - Income optimization suggestions
  - Dividend calendar with upcoming payments

## Setup Instructions

### Prerequisites
- Grafana server running (version 7.0+)
- Portfolio Analytics API service running
- Proper data source configuration

### Quick Setup
Use the provided setup script to create all dashboards:

```bash
# Set up dashboards for multiple portfolios
python scripts/setup_analytics_dashboards.py --portfolio-ids portfolio1 portfolio2 portfolio3

# Set up dashboards for a single portfolio
python scripts/setup_analytics_dashboards.py --single-portfolio my-portfolio

# Validate existing dashboard templates
python scripts/setup_analytics_dashboards.py --validate-only --portfolio-ids dummy
```

### Manual Setup

1. **Configure Data Sources**:
   ```bash
   cp monitoring/grafana/datasources/analytics-datasource.yml /etc/grafana/provisioning/datasources/
   ```

2. **Set up Dashboard Provisioning**:
   ```bash
   cp monitoring/grafana/provisioning/dashboards/comprehensive-dashboards.yml /etc/grafana/provisioning/dashboards/
   ```

3. **Copy Dashboard Files**:
   ```bash
   cp monitoring/grafana/dashboards/*.json /etc/grafana/provisioning/dashboards/
   ```

4. **Restart Grafana**:
   ```bash
   sudo systemctl restart grafana-server
   ```

## Dashboard Configuration

### Template Variables
Each dashboard includes template variables for customization:

- **Portfolio**: Select specific portfolio for analysis
- **Benchmark**: Choose benchmark for comparison (S&P 500, Total Market, etc.)
- **Time Period**: Adjust analysis time window
- **Confidence Level**: Set VaR confidence levels (risk dashboard)
- **Strategy**: Filter by optimization strategy (backtesting dashboard)

### Data Source Requirements
The dashboards expect the following data sources:

1. **Portfolio Analytics API** (Prometheus format):
   - URL: `http://analytics-service:8084`
   - Type: Prometheus
   - Authentication: Bearer token

2. **Portfolio Analytics JSON** (SimpleJSON format):
   - URL: `http://analytics-service:8084/api/v1/performance`
   - Type: SimpleJSON
   - Authentication: Bearer token

### Metric Naming Convention
The dashboards use standardized metric names:

- `portfolio_*`: Portfolio-specific metrics
- `benchmark_*`: Benchmark comparison metrics
- `backtest_*`: Backtesting-related metrics
- `risk_*`: Risk analysis metrics
- `dividend_*`: Dividend and income metrics

## Customization

### Adding New Panels
To add custom panels to existing dashboards:

1. Edit the JSON dashboard file
2. Add new panel configuration
3. Update panel IDs and grid positions
4. Restart Grafana or refresh provisioning

### Creating Custom Dashboards
Use the existing dashboards as templates:

1. Copy an existing dashboard JSON file
2. Modify title, panels, and queries
3. Update template variables as needed
4. Add to provisioning configuration

### Modifying Queries
Dashboard queries use Prometheus query language (PromQL):

```promql
# Example: Portfolio total return
portfolio_total_return{portfolio_id="$portfolio"}

# Example: Risk metrics with time range
portfolio_var_95{portfolio_id="$portfolio"}[30d]
```

## Troubleshooting

### Common Issues

1. **Dashboard Not Loading**:
   - Check data source connectivity
   - Verify metric names in queries
   - Check Grafana logs for errors

2. **No Data Displayed**:
   - Ensure Analytics API is running
   - Verify portfolio ID exists
   - Check time range settings

3. **Template Variables Not Working**:
   - Verify query syntax in template variables
   - Check data source configuration
   - Ensure metrics have proper labels

### Validation
Use the validation script to check dashboard integrity:

```bash
python scripts/setup_analytics_dashboards.py --validate-only --portfolio-ids dummy
```

### Logs
Check Grafana logs for dashboard-related issues:

```bash
sudo journalctl -u grafana-server -f
```

## Performance Considerations

### Dashboard Optimization
- Use appropriate time ranges to avoid large queries
- Implement query caching where possible
- Consider using recording rules for complex calculations
- Monitor dashboard load times

### Data Source Optimization
- Configure proper connection pooling
- Set appropriate query timeouts
- Use data source proxy for better performance
- Implement metric aggregation for historical data

## Security

### Access Control
- Configure proper user permissions in Grafana
- Use role-based access control (RBAC)
- Implement dashboard-level permissions
- Secure API endpoints with authentication

### Data Privacy
- Ensure sensitive portfolio data is properly secured
- Use encrypted connections (HTTPS/TLS)
- Implement audit logging for dashboard access
- Consider data anonymization for demo environments

## Maintenance

### Regular Tasks
- Update dashboard templates as needed
- Monitor dashboard performance
- Review and update alert thresholds
- Backup dashboard configurations

### Version Control
- Keep dashboard JSON files in version control
- Document changes and updates
- Test dashboard changes in staging environment
- Use proper deployment procedures

## Support

For issues or questions regarding the analytics dashboards:

1. Check the troubleshooting section above
2. Review Grafana and Analytics API logs
3. Validate dashboard templates using the provided script
4. Consult the Portfolio Analytics API documentation

## Contributing

When contributing new dashboards or improvements:

1. Follow the existing naming conventions
2. Include proper documentation
3. Test thoroughly with sample data
4. Update this README with new features
5. Ensure backward compatibility where possible