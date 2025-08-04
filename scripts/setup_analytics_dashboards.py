#!/usr/bin/env python3
"""
Script to set up comprehensive analytics dashboards for portfolio analytics.

This script creates and configures Grafana dashboards for:
- Portfolio performance analytics with benchmark comparisons
- Risk analytics with correlation heat maps and VaR metrics
- Backtesting results with strategy comparison charts
- Dividend income analytics with yield tracking and projections
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio_rebalancer.analytics.grafana_integration import (
    GrafanaDashboardProvisioner,
    setup_grafana_integration
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dashboards_for_portfolio(portfolio_id: str, 
                                  output_dir: str = "./monitoring/grafana/dashboards",
                                  backtest_id: Optional[str] = None) -> None:
    """
    Create all comprehensive dashboards for a specific portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
        output_dir: Output directory for dashboard files
        backtest_id: Optional backtest identifier
    """
    try:
        provisioner = GrafanaDashboardProvisioner()
        
        logger.info(f"Creating comprehensive dashboards for portfolio: {portfolio_id}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create all dashboard types
        dashboards = provisioner.create_all_dashboards(portfolio_id, backtest_id)
        
        # Export each dashboard
        for dashboard_type, dashboard_config in dashboards.items():
            filename = f"{dashboard_type}-{portfolio_id}.json"
            if dashboard_type == "backtesting" and backtest_id:
                filename = f"{dashboard_type}-{backtest_id}.json"
            
            file_path = output_path / filename
            with open(file_path, 'w') as f:
                json.dump(dashboard_config, f, indent=2)
            
            logger.info(f"Created {dashboard_type} dashboard: {file_path}")
        
        logger.info(f"Successfully created {len(dashboards)} dashboards for portfolio {portfolio_id}")
        
    except Exception as e:
        logger.error(f"Failed to create dashboards for portfolio {portfolio_id}: {e}")
        raise


def setup_complete_integration(portfolio_ids: List[str],
                             analytics_api_url: str = "http://analytics-service:8084",
                             grafana_config_path: str = "./monitoring/grafana") -> None:
    """
    Set up complete Grafana integration with all comprehensive dashboards.
    
    Args:
        portfolio_ids: List of portfolio identifiers
        analytics_api_url: Analytics API base URL
        grafana_config_path: Path to Grafana configuration directory
    """
    try:
        logger.info(f"Setting up complete Grafana integration for {len(portfolio_ids)} portfolios")
        
        # Use the existing setup function which now creates comprehensive dashboards
        setup_grafana_integration(portfolio_ids, analytics_api_url, grafana_config_path)
        
        logger.info("Complete Grafana integration setup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup complete Grafana integration: {e}")
        raise


def validate_dashboards(dashboard_dir: str = "./monitoring/grafana/dashboards") -> bool:
    """
    Validate that all required dashboard files exist and are valid JSON.
    
    Args:
        dashboard_dir: Directory containing dashboard files
        
    Returns:
        True if all dashboards are valid, False otherwise
    """
    try:
        dashboard_path = Path(dashboard_dir)
        
        if not dashboard_path.exists():
            logger.error(f"Dashboard directory does not exist: {dashboard_path}")
            return False
        
        # Required dashboard templates
        required_dashboards = [
            "portfolio-performance-comprehensive.json",
            "risk-analytics-dashboard.json", 
            "backtesting-results-dashboard.json",
            "dividend-income-dashboard.json"
        ]
        
        all_valid = True
        
        for dashboard_file in required_dashboards:
            file_path = dashboard_path / dashboard_file
            
            if not file_path.exists():
                logger.error(f"Required dashboard template missing: {file_path}")
                all_valid = False
                continue
            
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
                logger.info(f"Dashboard template valid: {dashboard_file}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in dashboard template {dashboard_file}: {e}")
                all_valid = False
        
        if all_valid:
            logger.info("All dashboard templates are valid")
        else:
            logger.error("Some dashboard templates are invalid or missing")
        
        return all_valid
        
    except Exception as e:
        logger.error(f"Failed to validate dashboards: {e}")
        return False


def main():
    """Main function to handle command line arguments and execute dashboard setup."""
    parser = argparse.ArgumentParser(
        description="Set up comprehensive analytics dashboards for portfolio analytics"
    )
    
    parser.add_argument(
        "--portfolio-ids",
        nargs="+",
        help="List of portfolio identifiers to create dashboards for"
    )
    
    parser.add_argument(
        "--analytics-api-url",
        default="http://analytics-service:8084",
        help="Analytics API base URL (default: http://analytics-service:8084)"
    )
    
    parser.add_argument(
        "--grafana-config-path",
        default="./monitoring/grafana",
        help="Path to Grafana configuration directory (default: ./monitoring/grafana)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./monitoring/grafana/dashboards",
        help="Output directory for dashboard files (default: ./monitoring/grafana/dashboards)"
    )
    
    parser.add_argument(
        "--backtest-id",
        help="Optional backtest identifier for backtesting dashboard"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing dashboard templates without creating new ones"
    )
    
    parser.add_argument(
        "--single-portfolio",
        help="Create dashboards for a single portfolio only"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate dashboards first (check templates in monitoring directory)
        template_dir = "./monitoring/grafana/dashboards"
        if not validate_dashboards(template_dir):
            logger.error("Dashboard validation failed")
            sys.exit(1)
        
        if args.validate_only:
            logger.info("Dashboard validation completed successfully")
            return
        
        # Create dashboards for single portfolio if specified
        if args.single_portfolio:
            create_dashboards_for_portfolio(
                args.single_portfolio,
                args.output_dir,
                args.backtest_id
            )
            return
        
        # Check if portfolio_ids is provided for complete integration
        if not args.portfolio_ids:
            logger.error("Either --single-portfolio or --portfolio-ids must be specified")
            sys.exit(1)
        
        # Set up complete integration for all portfolios
        setup_complete_integration(
            args.portfolio_ids,
            args.analytics_api_url,
            args.grafana_config_path
        )
        
        logger.info("Dashboard setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Dashboard setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()