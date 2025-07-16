"""Unit tests for YFinanceProvider."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import date, datetime, timedelta
import pandas as pd
from requests.exceptions import RequestException, HTTPError, Timeout, ConnectionError

from src.portfolio_rebalancer.fetcher.yfinance_provider import YFinanceProvider, YFinanceError


class TestYFinanceProvider:
    """Test cases for YFinanceProvider class."""
    
    @pytest.fixture
    def provider(self):
        """Create a YFinanceProvider instance for testing."""
        return YFinanceProvider(max_retries=2, base_delay=0.1, max_delay=1.0, timeout=10.0)
    
    @pytest.fixture
    def sample_yfinance_data(self):
        """Create sample yfinance historical data."""
        dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
        data = {
            'Adj Close': [100.0, 101.5, 99.8, 102.3, 103.1],
            'Volume': [1000000, 1200000, 800000, 1500000, 900000]
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Date'
        return df
    
    @pytest.fixture
    def expected_converted_data(self):
        """Create expected converted data format."""
        data = {
            'adjusted_close': [100.0, 101.5, 99.8, 102.3, 103.1],
            'volume': [1000000, 1200000, 800000, 1500000, 900000]
        }
        dates = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)]
        symbols = ['AAPL'] * 5
        
        df = pd.DataFrame(data)
        df['date'] = dates
        df['symbol'] = symbols
        df = df.set_index(['date', 'symbol'])
        return df
    
    def test_init_default_parameters(self):
        """Test YFinanceProvider initialization with default parameters."""
        provider = YFinanceProvider()
        assert provider.max_retries == 3
        assert provider.base_delay == 1.0
        assert provider.max_delay == 60.0
        assert provider.timeout == 30.0
    
    def test_init_custom_parameters(self):
        """Test YFinanceProvider initialization with custom parameters."""
        provider = YFinanceProvider(max_retries=5, base_delay=2.0, max_delay=120.0, timeout=45.0)
        assert provider.max_retries == 5
        assert provider.base_delay == 2.0
        assert provider.max_delay == 120.0
        assert provider.timeout == 45.0
    
    def test_fetch_prices_empty_tickers(self, provider):
        """Test fetch_prices with empty tickers list."""
        with pytest.raises(ValueError, match="Tickers list cannot be empty"):
            provider.fetch_prices([], date(2023, 1, 1), date(2023, 1, 5))
    
    def test_fetch_prices_invalid_date_range(self, provider):
        """Test fetch_prices with invalid date range."""
        with pytest.raises(ValueError, match="Start date cannot be after end date"):
            provider.fetch_prices(['AAPL'], date(2023, 1, 5), date(2023, 1, 1))
    
    def test_fetch_prices_whitespace_tickers(self, provider):
        """Test fetch_prices with whitespace-only tickers."""
        with pytest.raises(ValueError, match="No valid tickers provided"):
            provider.fetch_prices(['  ', '\t', ''], date(2023, 1, 1), date(2023, 1, 5))
    
    @patch('src.portfolio_rebalancer.fetcher.yfinance_provider.yf.Ticker')
    def test_fetch_prices_success_single_ticker(self, mock_ticker_class, provider, sample_yfinance_data, expected_converted_data):
        """Test successful fetch_prices for single ticker."""
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = sample_yfinance_data
        mock_ticker_class.return_value = mock_ticker
        
        # Execute
        result = provider.fetch_prices(['AAPL'], date(2023, 1, 1), date(2023, 1, 5))
        
        # Verify
        assert not result.empty
        assert list(result.columns) == ['adjusted_close', 'volume']
        assert len(result) == 5
        
        # Check that yfinance was called correctly
        mock_ticker_class.assert_called_once_with('AAPL')
        mock_ticker.history.assert_called_once_with(
            start=date(2023, 1, 1),
            end=date(2023, 1, 6),  # End date + 1 day
            timeout=10.0,
            raise_errors=True
        )
        
        # Verify data structure
        assert result.index.names == ['date', 'symbol']
        symbols = result.index.get_level_values('symbol').unique()
        assert 'AAPL' in symbols
    
    @patch('src.portfolio_rebalancer.fetcher.yfinance_provider.yf.Ticker')
    def test_fetch_prices_success_multiple_tickers(self, mock_ticker_class, provider):
        """Test successful fetch_prices for multiple tickers."""
        # Create sample data for two tickers
        dates = pd.date_range('2023-01-01', '2023-01-03', freq='D')
        
        aapl_data = pd.DataFrame({
            'Adj Close': [100.0, 101.0, 102.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=dates)
        aapl_data.index.name = 'Date'
        
        msft_data = pd.DataFrame({
            'Adj Close': [200.0, 201.0, 202.0],
            'Volume': [2000000, 2100000, 2200000]
        }, index=dates)
        msft_data.index.name = 'Date'
        
        # Setup mock to return different data for different tickers
        def mock_ticker_side_effect(ticker):
            mock_ticker = Mock()
            if ticker == 'AAPL':
                mock_ticker.history.return_value = aapl_data
            elif ticker == 'MSFT':
                mock_ticker.history.return_value = msft_data
            return mock_ticker
        
        mock_ticker_class.side_effect = mock_ticker_side_effect
        
        # Execute
        result = provider.fetch_prices(['AAPL', 'MSFT'], date(2023, 1, 1), date(2023, 1, 3))
        
        # Verify
        assert not result.empty
        assert len(result) == 6  # 3 days * 2 tickers
        symbols = result.index.get_level_values('symbol').unique()
        assert set(symbols) == {'AAPL', 'MSFT'}
        
        # Check AAPL data
        aapl_data_result = result.xs('AAPL', level='symbol')
        assert len(aapl_data_result) == 3
        assert aapl_data_result['adjusted_close'].iloc[0] == 100.0
        
        # Check MSFT data
        msft_data_result = result.xs('MSFT', level='symbol')
        assert len(msft_data_result) == 3
        assert msft_data_result['adjusted_close'].iloc[0] == 200.0
    
    @patch('src.portfolio_rebalancer.fetcher.yfinance_provider.yf.Ticker')
    def test_fetch_prices_empty_data_response(self, mock_ticker_class, provider):
        """Test fetch_prices when yfinance returns empty data."""
        # Setup mock to return empty DataFrame
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        # Execute and verify exception
        with pytest.raises(YFinanceError, match="Failed to fetch data for all tickers"):
            provider.fetch_prices(['INVALID'], date(2023, 1, 1), date(2023, 1, 5))
    
    @patch('src.portfolio_rebalancer.fetcher.yfinance_provider.yf.Ticker')
    @patch('src.portfolio_rebalancer.fetcher.yfinance_provider.time.sleep')
    def test_fetch_prices_network_error_with_retry(self, mock_sleep, mock_ticker_class, provider):
        """Test fetch_prices with network error and successful retry."""
        # Setup mock to fail first time, succeed second time
        mock_ticker = Mock()
        success_df = pd.DataFrame({
            'Adj Close': [100.0],
            'Volume': [1000000]
        }, index=pd.date_range('2023-01-01', periods=1))
        success_df.index.name = 'Date'
        
        mock_ticker.history.side_effect = [
            RequestException("Network error"),
            success_df
        ]
        mock_ticker_class.return_value = mock_ticker
        
        # Execute
        result = provider.fetch_prices(['AAPL'], date(2023, 1, 1), date(2023, 1, 1))
        
        # Verify retry occurred
        assert mock_ticker.history.call_count == 2
        mock_sleep.assert_called_once()
        assert not result.empty
    
    @patch('src.portfolio_rebalancer.fetcher.yfinance_provider.yf.Ticker')
    @patch('src.portfolio_rebalancer.fetcher.yfinance_provider.time.sleep')
    def test_fetch_prices_persistent_network_error(self, mock_sleep, mock_ticker_class, provider):
        """Test fetch_prices with persistent network error."""
        # Setup mock to always fail
        mock_ticker = Mock()
        mock_ticker.history.side_effect = RequestException("Persistent network error")
        mock_ticker_class.return_value = mock_ticker
        
        # Execute and verify exception
        with pytest.raises(YFinanceError, match="Failed to fetch data for all tickers"):
            provider.fetch_prices(['AAPL'], date(2023, 1, 1), date(2023, 1, 1))
        
        # Verify all retries were attempted
        assert mock_ticker.history.call_count == 3  # Initial + 2 retries
        assert mock_sleep.call_count == 2  # 2 retry delays
    
    @patch('src.portfolio_rebalancer.fetcher.yfinance_provider.yf.Ticker')
    def test_fetch_prices_partial_failure(self, mock_ticker_class, provider):
        """Test fetch_prices with some tickers failing."""
        # Setup mock to succeed for AAPL, fail for INVALID
        def mock_ticker_side_effect(ticker):
            mock_ticker = Mock()
            if ticker == 'AAPL':
                success_df = pd.DataFrame({
                    'Adj Close': [100.0],
                    'Volume': [1000000]
                }, index=pd.date_range('2023-01-01', periods=1))
                success_df.index.name = 'Date'
                mock_ticker.history.return_value = success_df
            else:
                mock_ticker.history.side_effect = Exception("Invalid ticker")
            return mock_ticker
        
        mock_ticker_class.side_effect = mock_ticker_side_effect
        
        # Execute
        result = provider.fetch_prices(['AAPL', 'INVALID'], date(2023, 1, 1), date(2023, 1, 1))
        
        # Verify partial success
        assert not result.empty
        assert len(result) == 1
        symbols = result.index.get_level_values('symbol').unique()
        assert list(symbols) == ['AAPL']
    
    def test_convert_yfinance_data(self, provider, sample_yfinance_data):
        """Test _convert_yfinance_data method."""
        result = provider._convert_yfinance_data(sample_yfinance_data, 'AAPL')
        
        # Verify structure
        assert result.index.names == ['date', 'symbol']
        assert list(result.columns) == ['adjusted_close', 'volume']
        assert len(result) == 5
        
        # Verify data
        assert result['adjusted_close'].iloc[0] == 100.0
        assert result['volume'].iloc[0] == 1000000
        
        # Verify all rows have correct symbol
        symbols = result.index.get_level_values('symbol').unique()
        assert list(symbols) == ['AAPL']
    
    def test_validate_and_clean_data_valid_data(self, provider):
        """Test _validate_and_clean_data with valid data."""
        # Create valid test data
        data = {
            'adjusted_close': [100.0, 101.5, 99.8],
            'volume': [1000000, 1200000, 800000]
        }
        dates = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)]
        symbols = ['AAPL'] * 3
        
        df = pd.DataFrame(data)
        df['date'] = dates
        df['symbol'] = symbols
        df = df.set_index(['date', 'symbol'])
        
        result = provider._validate_and_clean_data(df)
        
        # Should return same data since it's all valid
        assert len(result) == 3
        assert list(result.columns) == ['adjusted_close', 'volume']
    
    def test_validate_and_clean_data_invalid_prices(self, provider):
        """Test _validate_and_clean_data with invalid prices."""
        # Create data with invalid prices
        data = {
            'adjusted_close': [100.0, -50.0, 0.0, 101.5],  # Negative and zero prices
            'volume': [1000000, 1200000, 800000, 900000]
        }
        dates = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)]
        symbols = ['AAPL'] * 4
        
        df = pd.DataFrame(data)
        df['date'] = dates
        df['symbol'] = symbols
        df = df.set_index(['date', 'symbol'])
        
        result = provider._validate_and_clean_data(df)
        
        # Should only keep valid prices
        assert len(result) == 2  # Only positive prices
        assert all(result['adjusted_close'] > 0)
    
    def test_validate_and_clean_data_negative_volume(self, provider):
        """Test _validate_and_clean_data with negative volume."""
        # Create data with negative volume
        data = {
            'adjusted_close': [100.0, 101.5],
            'volume': [1000000, -500000]  # Negative volume
        }
        dates = [date(2023, 1, 1), date(2023, 1, 2)]
        symbols = ['AAPL'] * 2
        
        df = pd.DataFrame(data)
        df['date'] = dates
        df['symbol'] = symbols
        df = df.set_index(['date', 'symbol'])
        
        result = provider._validate_and_clean_data(df)
        
        # Should only keep non-negative volume
        assert len(result) == 1
        assert all(result['volume'] >= 0)
    
    def test_validate_and_clean_data_empty_dataframe(self, provider):
        """Test _validate_and_clean_data with empty DataFrame."""
        df = pd.DataFrame()
        result = provider._validate_and_clean_data(df)
        assert result.empty
    
    @patch('src.portfolio_rebalancer.fetcher.yfinance_provider.yf.Ticker')
    def test_ticker_case_normalization(self, mock_ticker_class, provider):
        """Test that ticker symbols are normalized to uppercase."""
        mock_ticker = Mock()
        success_df = pd.DataFrame({
            'Adj Close': [100.0],
            'Volume': [1000000]
        }, index=pd.date_range('2023-01-01', periods=1))
        success_df.index.name = 'Date'
        mock_ticker.history.return_value = success_df
        mock_ticker_class.return_value = mock_ticker
        
        # Test with lowercase ticker
        result = provider.fetch_prices(['aapl'], date(2023, 1, 1), date(2023, 1, 1))
        
        # Verify ticker was normalized to uppercase
        mock_ticker_class.assert_called_once_with('AAPL')
        symbols = result.index.get_level_values('symbol').unique()
        assert 'AAPL' in symbols
    
    @patch('src.portfolio_rebalancer.fetcher.yfinance_provider.yf.Ticker')
    def test_exponential_backoff_delay_calculation(self, mock_ticker_class, provider):
        """Test that exponential backoff delays are calculated correctly."""
        with patch('src.portfolio_rebalancer.fetcher.yfinance_provider.time.sleep') as mock_sleep:
            # Setup mock to fail multiple times
            mock_ticker = Mock()
            mock_ticker.history.side_effect = [
                RequestException("Error 1"),
                RequestException("Error 2"),
                RequestException("Error 3")
            ]
            mock_ticker_class.return_value = mock_ticker
            
            # Execute (should fail after all retries)
            with pytest.raises(YFinanceError):
                provider.fetch_prices(['AAPL'], date(2023, 1, 1), date(2023, 1, 1))
            
            # Verify exponential backoff delays
            expected_delays = [0.1, 0.2]  # base_delay * 2^attempt, capped by max_delay
            actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
            assert actual_delays == expected_delays
    
    def test_data_quality_validation_extreme_price_changes(self, provider):
        """Test data quality validation for extreme price changes."""
        # Create data with extreme price change (>50%)
        data = {
            'adjusted_close': [100.0, 200.0, 90.0],  # 100% increase, then 55% decrease
            'volume': [1000000, 1200000, 800000]
        }
        dates = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)]
        symbols = ['AAPL'] * 3
        
        df = pd.DataFrame(data)
        df['date'] = dates
        df['symbol'] = symbols
        df = df.set_index(['date', 'symbol'])
        
        # Should not raise exception but log warning
        with patch('src.portfolio_rebalancer.fetcher.yfinance_provider.logger') as mock_logger:
            result = provider._validate_and_clean_data(df)
            
            # Data should still be included (we only warn, don't remove)
            assert len(result) == 3
            
            # Should have logged warning about extreme changes
            mock_logger.warning.assert_called()
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if 'Extreme price changes detected' in str(call)]
            assert len(warning_calls) > 0