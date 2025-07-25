-- Analytics Database Migration Scripts
-- Version: 1.0.0
-- Description: Create analytics tables for portfolio analytics service

-- Enable UUID extension if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create analytics schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set search path
SET search_path TO analytics, public;

-- ============================================================================
-- Backtest Results Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    config_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Configuration
    tickers JSONB NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    rebalance_frequency VARCHAR(20) NOT NULL DEFAULT 'monthly',
    transaction_cost DECIMAL(8,6) NOT NULL DEFAULT 0.001,
    initial_capital DECIMAL(15,2) NOT NULL DEFAULT 100000.00,
    
    -- Results
    total_return DECIMAL(10,6),
    annualized_return DECIMAL(10,6),
    volatility DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    calmar_ratio DECIMAL(10,6),
    transaction_costs DECIMAL(15,2),
    num_rebalances INTEGER,
    final_value DECIMAL(15,2),
    
    -- Serialized time series data
    returns_data JSONB,
    allocation_data JSONB,
    
    -- Constraints
    CONSTRAINT chk_backtest_dates CHECK (start_date < end_date),
    CONSTRAINT chk_backtest_transaction_cost CHECK (transaction_cost >= 0 AND transaction_cost <= 0.1),
    CONSTRAINT chk_backtest_initial_capital CHECK (initial_capital > 0),
    CONSTRAINT chk_backtest_strategy CHECK (strategy IN ('sharpe', 'min_variance', 'equal_weight', 'risk_parity')),
    CONSTRAINT chk_backtest_frequency CHECK (rebalance_frequency IN ('daily', 'weekly', 'monthly', 'quarterly'))
);

-- Indexes for backtest_results
CREATE INDEX IF NOT EXISTS idx_backtest_config_hash ON backtest_results(config_hash, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_dates ON backtest_results(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_backtest_created_at ON backtest_results(created_at DESC);

-- ============================================================================
-- Monte Carlo Results Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS monte_carlo_results (
    id SERIAL PRIMARY KEY,
    portfolio_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Configuration
    portfolio_tickers JSONB NOT NULL,
    portfolio_weights JSONB NOT NULL,
    time_horizon_years INTEGER NOT NULL,
    num_simulations INTEGER NOT NULL DEFAULT 10000,
    initial_value DECIMAL(15,2) NOT NULL DEFAULT 100000.00,
    confidence_levels JSONB NOT NULL DEFAULT '[0.05, 0.25, 0.5, 0.75, 0.95]',
    
    -- Results
    expected_value DECIMAL(15,2),
    probability_of_loss DECIMAL(8,6),
    value_at_risk_95 DECIMAL(15,2),
    conditional_var_95 DECIMAL(15,2),
    
    -- Serialized simulation data
    percentile_data JSONB,
    simulation_summary JSONB,
    
    -- Constraints
    CONSTRAINT chk_monte_carlo_horizon CHECK (time_horizon_years > 0 AND time_horizon_years <= 50),
    CONSTRAINT chk_monte_carlo_simulations CHECK (num_simulations > 0 AND num_simulations <= 100000),
    CONSTRAINT chk_monte_carlo_initial_value CHECK (initial_value > 0),
    CONSTRAINT chk_monte_carlo_prob_loss CHECK (probability_of_loss >= 0 AND probability_of_loss <= 1)
);

-- Indexes for monte_carlo_results
CREATE INDEX IF NOT EXISTS idx_monte_carlo_portfolio ON monte_carlo_results(portfolio_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_monte_carlo_horizon ON monte_carlo_results(time_horizon_years, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_monte_carlo_created_at ON monte_carlo_results(created_at DESC);

-- ============================================================================
-- Risk Analysis Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS risk_analysis (
    id SERIAL PRIMARY KEY,
    portfolio_id VARCHAR(100) NOT NULL,
    analysis_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Risk metrics
    portfolio_beta DECIMAL(8,4),
    tracking_error DECIMAL(10,6),
    information_ratio DECIMAL(10,6),
    var_95 DECIMAL(10,6),
    cvar_95 DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    concentration_risk DECIMAL(8,6),
    
    -- Serialized complex data
    correlation_data JSONB,
    factor_exposures JSONB,
    sector_exposures JSONB,
    
    -- Constraints
    CONSTRAINT chk_risk_tracking_error CHECK (tracking_error >= 0),
    CONSTRAINT chk_risk_max_drawdown CHECK (max_drawdown <= 0),
    CONSTRAINT chk_risk_concentration CHECK (concentration_risk >= 0 AND concentration_risk <= 1)
);

-- Unique constraint and indexes for risk_analysis
CREATE UNIQUE INDEX IF NOT EXISTS uq_risk_portfolio_date ON risk_analysis(portfolio_id, analysis_date);
CREATE INDEX IF NOT EXISTS idx_risk_portfolio ON risk_analysis(portfolio_id, analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_risk_created_at ON risk_analysis(created_at DESC);

-- ============================================================================
-- Performance Metrics Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    portfolio_id VARCHAR(100) NOT NULL,
    calculation_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Performance metrics
    total_return DECIMAL(10,6),
    annualized_return DECIMAL(10,6),
    volatility DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    sortino_ratio DECIMAL(10,6),
    alpha DECIMAL(10,6),
    beta DECIMAL(8,4),
    r_squared DECIMAL(8,6),
    tracking_error DECIMAL(10,6),
    information_ratio DECIMAL(10,6),
    
    -- Serialized time series data
    performance_data JSONB,
    
    -- Constraints
    CONSTRAINT chk_perf_volatility CHECK (volatility >= 0),
    CONSTRAINT chk_perf_r_squared CHECK (r_squared >= 0 AND r_squared <= 1),
    CONSTRAINT chk_perf_tracking_error CHECK (tracking_error >= 0)
);

-- Unique constraint and indexes for performance_metrics
CREATE UNIQUE INDEX IF NOT EXISTS uq_perf_portfolio_date ON performance_metrics(portfolio_id, calculation_date);
CREATE INDEX IF NOT EXISTS idx_perf_portfolio ON performance_metrics(portfolio_id, calculation_date DESC);
CREATE INDEX IF NOT EXISTS idx_perf_created_at ON performance_metrics(created_at DESC);

-- ============================================================================
-- Dividend Analysis Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS dividend_analysis (
    id SERIAL PRIMARY KEY,
    portfolio_id VARCHAR(100) NOT NULL,
    analysis_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Dividend metrics
    current_yield DECIMAL(8,6),
    projected_annual_income DECIMAL(15,2),
    dividend_growth_rate DECIMAL(8,6),
    payout_ratio DECIMAL(8,6),
    dividend_coverage DECIMAL(8,4),
    income_sustainability_score DECIMAL(4,3),
    
    -- Serialized dividend data
    dividend_data JSONB,
    top_contributors JSONB,
    
    -- Constraints
    CONSTRAINT chk_div_current_yield CHECK (current_yield >= 0),
    CONSTRAINT chk_div_projected_income CHECK (projected_annual_income >= 0),
    CONSTRAINT chk_div_payout_ratio CHECK (payout_ratio >= 0),
    CONSTRAINT chk_div_coverage CHECK (dividend_coverage >= 0),
    CONSTRAINT chk_div_sustainability CHECK (income_sustainability_score >= 0 AND income_sustainability_score <= 1)
);

-- Unique constraint and indexes for dividend_analysis
CREATE UNIQUE INDEX IF NOT EXISTS uq_div_portfolio_date ON dividend_analysis(portfolio_id, analysis_date);
CREATE INDEX IF NOT EXISTS idx_div_portfolio ON dividend_analysis(portfolio_id, analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_div_created_at ON dividend_analysis(created_at DESC);

-- ============================================================================
-- Additional Analytics Tables
-- ============================================================================

-- Strategy Comparison Results Table
CREATE TABLE IF NOT EXISTS strategy_comparisons (
    id SERIAL PRIMARY KEY,
    comparison_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Comparison configuration
    strategies JSONB NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    tickers JSONB NOT NULL,
    
    -- Results
    best_strategy VARCHAR(50),
    strategy_ranking JSONB,
    statistical_significance JSONB,
    comparison_results JSONB,
    
    -- Constraints
    CONSTRAINT chk_strategy_comp_dates CHECK (start_date < end_date)
);

CREATE INDEX IF NOT EXISTS idx_strategy_comp_created_at ON strategy_comparisons(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_comp_dates ON strategy_comparisons(start_date, end_date);

-- Stress Test Results Table
CREATE TABLE IF NOT EXISTS stress_test_results (
    id SERIAL PRIMARY KEY,
    portfolio_id VARCHAR(100) NOT NULL,
    test_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Test configuration and results
    scenarios JSONB NOT NULL,
    scenario_results JSONB NOT NULL,
    worst_case_loss DECIMAL(10,6),
    recovery_time_estimate INTEGER,
    
    -- Constraints
    CONSTRAINT chk_stress_worst_case CHECK (worst_case_loss <= 0)
);

CREATE INDEX IF NOT EXISTS idx_stress_test_portfolio ON stress_test_results(portfolio_id, test_date DESC);
CREATE INDEX IF NOT EXISTS idx_stress_test_created_at ON stress_test_results(created_at DESC);

-- VaR Calculation Results Table
CREATE TABLE IF NOT EXISTS var_results (
    id SERIAL PRIMARY KEY,
    portfolio_id VARCHAR(100) NOT NULL,
    calculation_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- VaR configuration
    confidence_level DECIMAL(4,3) NOT NULL,
    time_horizon_days INTEGER NOT NULL,
    methodology VARCHAR(50) NOT NULL DEFAULT 'historical',
    
    -- Results
    var_amount DECIMAL(15,2),
    cvar_amount DECIMAL(15,2),
    
    -- Constraints
    CONSTRAINT chk_var_confidence CHECK (confidence_level > 0 AND confidence_level < 1),
    CONSTRAINT chk_var_horizon CHECK (time_horizon_days > 0),
    CONSTRAINT chk_var_methodology CHECK (methodology IN ('historical', 'parametric', 'monte_carlo'))
);

CREATE INDEX IF NOT EXISTS idx_var_portfolio ON var_results(portfolio_id, calculation_date DESC);
CREATE INDEX IF NOT EXISTS idx_var_confidence ON var_results(confidence_level, calculation_date DESC);

-- ============================================================================
-- Views for Analytics Reporting
-- ============================================================================

-- Latest Risk Analysis View
CREATE OR REPLACE VIEW latest_risk_analysis AS
SELECT DISTINCT ON (portfolio_id) 
    portfolio_id,
    analysis_date,
    portfolio_beta,
    tracking_error,
    information_ratio,
    var_95,
    cvar_95,
    max_drawdown,
    concentration_risk,
    factor_exposures,
    sector_exposures,
    created_at
FROM risk_analysis
ORDER BY portfolio_id, analysis_date DESC, created_at DESC;

-- Latest Performance Metrics View
CREATE OR REPLACE VIEW latest_performance_metrics AS
SELECT DISTINCT ON (portfolio_id)
    portfolio_id,
    calculation_date,
    total_return,
    annualized_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    alpha,
    beta,
    r_squared,
    tracking_error,
    information_ratio,
    created_at
FROM performance_metrics
ORDER BY portfolio_id, calculation_date DESC, created_at DESC;

-- Latest Dividend Analysis View
CREATE OR REPLACE VIEW latest_dividend_analysis AS
SELECT DISTINCT ON (portfolio_id)
    portfolio_id,
    analysis_date,
    current_yield,
    projected_annual_income,
    dividend_growth_rate,
    payout_ratio,
    dividend_coverage,
    income_sustainability_score,
    top_contributors,
    created_at
FROM dividend_analysis
ORDER BY portfolio_id, analysis_date DESC, created_at DESC;

-- ============================================================================
-- Functions for Analytics Operations
-- ============================================================================

-- Function to clean old backtest results (keep last 100 per config_hash)
CREATE OR REPLACE FUNCTION cleanup_old_backtest_results()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    WITH ranked_results AS (
        SELECT id, 
               ROW_NUMBER() OVER (PARTITION BY config_hash ORDER BY created_at DESC) as rn
        FROM backtest_results
    )
    DELETE FROM backtest_results 
    WHERE id IN (
        SELECT id FROM ranked_results WHERE rn > 100
    );
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get portfolio performance summary
CREATE OR REPLACE FUNCTION get_portfolio_summary(p_portfolio_id VARCHAR(100))
RETURNS TABLE (
    portfolio_id VARCHAR(100),
    latest_performance_date DATE,
    latest_risk_date DATE,
    latest_dividend_date DATE,
    sharpe_ratio DECIMAL(10,6),
    var_95 DECIMAL(10,6),
    current_yield DECIMAL(8,6)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p_portfolio_id,
        pm.calculation_date,
        ra.analysis_date,
        da.analysis_date,
        pm.sharpe_ratio,
        ra.var_95,
        da.current_yield
    FROM latest_performance_metrics pm
    FULL OUTER JOIN latest_risk_analysis ra ON pm.portfolio_id = ra.portfolio_id
    FULL OUTER JOIN latest_dividend_analysis da ON pm.portfolio_id = da.portfolio_id
    WHERE COALESCE(pm.portfolio_id, ra.portfolio_id, da.portfolio_id) = p_portfolio_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Grants and Permissions
-- ============================================================================

-- Grant usage on schema
GRANT USAGE ON SCHEMA analytics TO PUBLIC;

-- Grant permissions on tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA analytics TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO PUBLIC;

-- Grant permissions on views
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO PUBLIC;

-- ============================================================================
-- Migration Completion
-- ============================================================================

-- Insert migration record
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    description TEXT
);

INSERT INTO schema_migrations (version, description) 
VALUES ('1.0.0', 'Initial analytics tables and schema creation')
ON CONFLICT (version) DO NOTHING;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Analytics database migration 1.0.0 completed successfully at %', CURRENT_TIMESTAMP;
END $$;