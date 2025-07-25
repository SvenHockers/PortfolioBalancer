-- Portfolio Rebalancer Database Initialization Script
-- This script creates the necessary database schema for the portfolio rebalancer system

-- Create database if it doesn't exist (this is handled by Docker environment variables)
-- CREATE DATABASE IF NOT EXISTS portfolio_rebalancer;

-- Use the database
-- \c portfolio_rebalancer;

-- Create analytics schema for analytics-specific tables
CREATE SCHEMA IF NOT EXISTS analytics;

-- Grant permissions to the portfolio user
GRANT ALL PRIVILEGES ON SCHEMA analytics TO portfolio_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO portfolio_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT ALL ON TABLES TO portfolio_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO portfolio_user;

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Analytics tables will be created by SQLAlchemy when the analytics service starts
-- This ensures proper table creation with all constraints and indexes

-- Create a simple health check table
CREATE TABLE IF NOT EXISTS health_check (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'healthy'
);

-- Insert initial health check record
INSERT INTO health_check (service_name, status) 
VALUES ('database', 'initialized') 
ON CONFLICT DO NOTHING;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_health_check_service ON health_check(service_name);
CREATE INDEX IF NOT EXISTS idx_health_check_timestamp ON health_check(last_check);

-- Log the initialization
INSERT INTO health_check (service_name, status) 
VALUES ('initialization', 'completed');