"""Comprehensive data export and integration utilities for analytics results."""

import json
import csv
import io
import gzip
import asyncio
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple, Union, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import requests
from urllib.parse import urljoin

from .models import (
    MonteCarloResult, MonteCarloConfig, BacktestResult, BacktestConfig,
    RiskAnalysis, PerformanceMetrics, DividendAnalysis
)
from .exceptions import ExportError, AnalyticsError

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    PDF = "pdf"


class CompressionType(str, Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    BZIP2 = "bzip2"


@dataclass
class ExportConfig:
    """Configuration for data export."""
    
    format: ExportFormat
    include_raw_data: bool = False
    include_statistics: bool = True
    include_percentiles: bool = True
    include_metadata: bool = True
    compression: Optional[CompressionType] = None
    decimal_places: int = 4
    page_size: Optional[int] = None  # For pagination
    streaming: bool = False  # For streaming large datasets
    
    def __post_init__(self):
        if isinstance(self.format, str):
            self.format = ExportFormat(self.format.lower())
        if isinstance(self.compression, str) and self.compression:
            self.compression = CompressionType(self.compression.lower())


@dataclass
class WebhookConfig:
    """Configuration for webhook notifications."""
    
    url: str
    method: str = "POST"
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Type": "application/json"}


@dataclass
class PaginationConfig:
    """Configuration for paginated exports."""
    
    page_size: int = 1000
    max_pages: Optional[int] = None
    include_total_count: bool = True
    
    def __post_init__(self):
        if self.page_size <= 0:
            raise ValueError("Page size must be positive")


@dataclass
class StreamingConfig:
    """Configuration for streaming exports."""
    
    chunk_size: int = 1000
    buffer_size: int = 8192
    compression: Optional[CompressionType] = None
    
    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")


class AnalyticsExporter:
    """Comprehensive exporter for all analytics results in various formats."""
    
    def __init__(self):
        """Initialize analytics exporter."""
        self.supported_formats = [f.value for f in ExportFormat]
        self.supported_result_types = [
            'backtest', 'monte_carlo', 'risk_analysis', 
            'performance', 'dividend_analysis'
        ]
    
    def export_result(self, 
                     result: Union[MonteCarloResult, BacktestResult, RiskAnalysis, 
                                 PerformanceMetrics, DividendAnalysis], 
                     config: ExportConfig) -> Tuple[bytes, str, str]:
        """
        Export analytics result in specified format.
        
        Args:
            result: Analytics result to export
            config: Export configuration
            
        Returns:
            Tuple of (data_bytes, content_type, filename)
        """
        try:
            logger.info(f"Exporting {type(result).__name__} in {config.format} format")
            
            if config.format == ExportFormat.JSON:
                return self._export_json(result, config)
            elif config.format == ExportFormat.CSV:
                return self._export_csv(result, config)
            elif config.format == ExportFormat.EXCEL:
                return self._export_excel(result, config)
            elif config.format == ExportFormat.PARQUET:
                return self._export_parquet(result, config)
            elif config.format == ExportFormat.HDF5:
                return self._export_hdf5(result, config)
            elif config.format == ExportFormat.PDF:
                return self._export_pdf(result, config)
            else:
                raise ExportError(f"Unsupported export format: {config.format}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise ExportError(f"Export failed: {str(e)}")
    
    async def export_result_streaming(self, 
                                    result: Union[MonteCarloResult, BacktestResult, RiskAnalysis, 
                                                PerformanceMetrics, DividendAnalysis],
                                    config: ExportConfig,
                                    streaming_config: StreamingConfig) -> AsyncGenerator[bytes, None]:
        """
        Export analytics result with streaming support for large datasets.
        
        Args:
            result: Analytics result to export
            config: Export configuration
            streaming_config: Streaming configuration
            
        Yields:
            Chunks of exported data
        """
        try:
            logger.info(f"Starting streaming export of {type(result).__name__}")
            
            if config.format == ExportFormat.JSON:
                async for chunk in self._export_json_streaming(result, config, streaming_config):
                    yield chunk
            elif config.format == ExportFormat.CSV:
                async for chunk in self._export_csv_streaming(result, config, streaming_config):
                    yield chunk
            else:
                # For non-streaming formats, export normally and yield in chunks
                data_bytes, _, _ = self.export_result(result, config)
                chunk_size = streaming_config.chunk_size
                
                for i in range(0, len(data_bytes), chunk_size):
                    yield data_bytes[i:i + chunk_size]
                    await asyncio.sleep(0)  # Allow other tasks to run
                    
        except Exception as e:
            logger.error(f"Streaming export failed: {e}")
            raise ExportError(f"Streaming export failed: {str(e)}")
    
    def export_multiple_results(self, 
                              results: List[Union[MonteCarloResult, BacktestResult, RiskAnalysis, 
                                                PerformanceMetrics, DividendAnalysis]], 
                              config: ExportConfig,
                              result_names: Optional[List[str]] = None,
                              pagination_config: Optional[PaginationConfig] = None) -> Tuple[bytes, str, str]:
        """
        Export multiple analytics results with optional pagination.
        
        Args:
            results: List of analytics results
            config: Export configuration
            result_names: Optional names for each result
            pagination_config: Optional pagination configuration
            
        Returns:
            Tuple of (data_bytes, content_type, filename)
        """
        try:
            if not results:
                raise ExportError("No results provided for export")
            
            if result_names and len(result_names) != len(results):
                raise ExportError("Number of result names must match number of results")
            
            if not result_names:
                result_names = [f"result_{i+1}" for i in range(len(results))]
            
            # Apply pagination if configured
            if pagination_config:
                results, result_names = self._apply_pagination(
                    results, result_names, pagination_config
                )
            
            logger.info(f"Exporting {len(results)} results in {config.format} format")
            
            if config.format == ExportFormat.EXCEL:
                return self._export_multiple_excel(results, result_names, config)
            elif config.format == ExportFormat.JSON:
                return self._export_multiple_json(results, result_names, config)
            elif config.format == ExportFormat.CSV:
                return self._export_multiple_csv(results, result_names, config)
            else:
                raise ExportError(f"Batch export not supported for format: {config.format}")
                
        except Exception as e:
            logger.error(f"Multiple results export failed: {e}")
            raise ExportError(f"Multiple results export failed: {str(e)}")
    
    def _export_json(self, 
                    result: Union[MonteCarloResult, BacktestResult, RiskAnalysis, 
                                PerformanceMetrics, DividendAnalysis], 
                    config: ExportConfig) -> Tuple[bytes, str, str]:
        """Export result as JSON."""
        
        # Prepare data structure
        export_data = self._prepare_export_data(result, config)
        
        # Convert to JSON
        json_str = json.dumps(
            export_data, 
            indent=2 if not config.compression else None,
            default=self._json_serializer
        )
        
        # Apply compression if requested
        data_bytes, filename = self._apply_compression(
            json_str.encode('utf-8'), 
            f"{self._get_result_type(result)}_result_{self._get_timestamp()}.json",
            config.compression
        )
        
        return data_bytes, "application/json", filename
    
    async def _export_json_streaming(self, 
                                   result: Union[MonteCarloResult, BacktestResult, RiskAnalysis, 
                                               PerformanceMetrics, DividendAnalysis],
                                   config: ExportConfig,
                                   streaming_config: StreamingConfig) -> AsyncGenerator[bytes, None]:
        """Export result as JSON with streaming support."""
        
        # Start JSON structure
        yield b'{\n'
        
        # Export metadata
        metadata = {
            "export_metadata": {
                "export_timestamp": datetime.utcnow().isoformat(),
                "export_format": config.format.value,
                "result_type": self._get_result_type(result),
                "streaming": True,
                "exporter_version": "2.0.0"
            }
        }
        
        metadata_json = json.dumps(metadata, indent=2, default=self._json_serializer)[1:-1]  # Remove outer braces
        yield metadata_json.encode('utf-8')
        yield b',\n'
        
        # Export main result data in chunks
        result_data = self._prepare_result_data(result, config)
        
        yield b'"result_data": {\n'
        
        items = list(result_data.items())
        for i, (key, value) in enumerate(items):
            key_value_json = json.dumps({key: value}, indent=2, default=self._json_serializer)[1:-1]
            yield key_value_json.encode('utf-8')
            
            if i < len(items) - 1:
                yield b',\n'
            
            await asyncio.sleep(0)  # Allow other tasks to run
        
        yield b'\n}\n'
        yield b'}'
    
    def _export_csv(self, 
                   result: Union[MonteCarloResult, BacktestResult, RiskAnalysis, 
                               PerformanceMetrics, DividendAnalysis], 
                   config: ExportConfig) -> Tuple[bytes, str, str]:
        """Export result as CSV."""
        
        # Create main results DataFrame based on result type
        df = self._create_result_dataframe(result, config)
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # Add additional data sections for specific result types
        if isinstance(result, MonteCarloResult) and config.include_percentiles and result.percentile_data:
            csv_buffer.write('\n\n# Percentile Data\n')
            percentile_df = self._create_percentile_dataframe(result.percentile_data, config)
            percentile_df.to_csv(csv_buffer, index=False)
        
        if isinstance(result, BacktestResult) and config.include_raw_data and result.returns_data:
            csv_buffer.write('\n\n# Returns Data\n')
            returns_df = self._create_returns_dataframe(result.returns_data, config)
            returns_df.to_csv(csv_buffer, index=False)
        
        data_bytes = csv_buffer.getvalue().encode('utf-8')
        
        # Apply compression if requested
        data_bytes, filename = self._apply_compression(
            data_bytes,
            f"{self._get_result_type(result)}_result_{self._get_timestamp()}.csv",
            config.compression
        )
        
        return data_bytes, "text/csv", filename
    
    async def _export_csv_streaming(self, 
                                  result: Union[MonteCarloResult, BacktestResult, RiskAnalysis, 
                                              PerformanceMetrics, DividendAnalysis],
                                  config: ExportConfig,
                                  streaming_config: StreamingConfig) -> AsyncGenerator[bytes, None]:
        """Export result as CSV with streaming support."""
        
        # Create main results DataFrame
        df = self._create_result_dataframe(result, config)
        
        # Stream CSV header
        csv_buffer = io.StringIO()
        df.head(0).to_csv(csv_buffer, index=False)  # Just headers
        yield csv_buffer.getvalue().encode('utf-8')
        
        # Stream data in chunks
        chunk_size = streaming_config.chunk_size
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            
            csv_buffer = io.StringIO()
            chunk_df.to_csv(csv_buffer, index=False, header=False)
            
            chunk_data = csv_buffer.getvalue().encode('utf-8')
            
            if streaming_config.compression == CompressionType.GZIP:
                chunk_data = gzip.compress(chunk_data)
            
            yield chunk_data
            await asyncio.sleep(0)  # Allow other tasks to run
    
    def _export_excel(self, 
                     result: Union[MonteCarloResult, BacktestResult, RiskAnalysis, 
                                 PerformanceMetrics, DividendAnalysis], 
                     config: ExportConfig) -> Tuple[bytes, str, str]:
        """Export result as Excel workbook."""
        
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            
            # Summary sheet
            summary_df = self._create_result_dataframe(result, config)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Configuration/metadata sheet
            if config.include_metadata:
                metadata_df = self._create_metadata_dataframe(result)
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Type-specific sheets
            if isinstance(result, MonteCarloResult):
                if config.include_statistics and result.simulation_summary:
                    stats_df = self._create_statistics_dataframe(result.simulation_summary, config)
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                if config.include_percentiles and result.percentile_data:
                    percentile_df = self._create_percentile_dataframe(result.percentile_data, config)
                    percentile_df.to_excel(writer, sheet_name='Percentiles', index=False)
                
                risk_df = self._create_risk_metrics_dataframe(result, config)
                risk_df.to_excel(writer, sheet_name='Risk_Metrics', index=False)
            
            elif isinstance(result, BacktestResult):
                if config.include_raw_data and result.returns_data:
                    returns_df = self._create_returns_dataframe(result.returns_data, config)
                    returns_df.to_excel(writer, sheet_name='Returns', index=False)
                
                if config.include_raw_data and result.allocation_data:
                    allocation_df = self._create_allocation_dataframe(result.allocation_data, config)
                    allocation_df.to_excel(writer, sheet_name='Allocations', index=False)
            
            elif isinstance(result, RiskAnalysis):
                if result.correlation_data:
                    corr_df = self._create_correlation_dataframe(result.correlation_data, config)
                    corr_df.to_excel(writer, sheet_name='Correlations', index=False)
                
                if result.factor_exposures:
                    factor_df = self._create_factor_exposure_dataframe(result.factor_exposures, config)
                    factor_df.to_excel(writer, sheet_name='Factor_Exposures', index=False)
            
            elif isinstance(result, DividendAnalysis):
                if result.dividend_data:
                    dividend_df = self._create_dividend_detail_dataframe(result.dividend_data, config)
                    dividend_df.to_excel(writer, sheet_name='Dividend_Details', index=False)
        
        data_bytes = excel_buffer.getvalue()
        
        # Apply compression if requested
        data_bytes, filename = self._apply_compression(
            data_bytes,
            f"{self._get_result_type(result)}_result_{self._get_timestamp()}.xlsx",
            config.compression
        )
        
        return data_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename
    
    def _export_pdf(self, 
                   result: Union[MonteCarloResult, BacktestResult, RiskAnalysis, 
                               PerformanceMetrics, DividendAnalysis], 
                   config: ExportConfig) -> Tuple[bytes, str, str]:
        """Export result as PDF report."""
        
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            
            result_type = self._get_result_type(result).replace('_', ' ').title()
            title = Paragraph(f"{result_type} Report", title_style)
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Metadata section
            story.append(Paragraph("Report Information", styles['Heading2']))
            metadata_data = [
                ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Result Type', result_type],
                ['Export Format', 'PDF'],
            ]
            
            if hasattr(result, 'config'):
                if hasattr(result.config, 'tickers'):
                    metadata_data.append(['Tickers', ', '.join(result.config.tickers)])
                if hasattr(result.config, 'portfolio_tickers'):
                    metadata_data.append(['Portfolio Tickers', ', '.join(result.config.portfolio_tickers)])
            
            metadata_table = Table(metadata_data)
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metadata_table)
            story.append(Spacer(1, 12))
            
            # Main results section
            story.append(Paragraph("Key Results", styles['Heading2']))
            
            # Create results table based on result type
            results_df = self._create_result_dataframe(result, config)
            results_data = [['Metric', 'Value']]
            
            for _, row in results_df.iterrows():
                if len(row) >= 2:
                    results_data.append([str(row.iloc[0]), str(row.iloc[1])])
            
            results_table = Table(results_data)
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(results_table)
            
            # Build PDF
            doc.build(story)
            data_bytes = pdf_buffer.getvalue()
            
            filename = f"{self._get_result_type(result)}_report_{self._get_timestamp()}.pdf"
            
            return data_bytes, "application/pdf", filename
            
        except ImportError:
            raise ExportError("PDF export requires reportlab package. Install with: pip install reportlab")
        except Exception as e:
            raise ExportError(f"PDF export failed: {str(e)}")
    
    def _export_parquet(self, 
                       result: MonteCarloResult, 
                       config: ExportConfig) -> Tuple[bytes, str, str]:
        """Export result as Parquet file."""
        
        # Create comprehensive DataFrame
        export_df = self._create_comprehensive_dataframe(result, config)
        
        parquet_buffer = io.BytesIO()
        export_df.to_parquet(parquet_buffer, index=False, compression='snappy')
        
        data_bytes = parquet_buffer.getvalue()
        filename = f"monte_carlo_result_{self._get_timestamp()}.parquet"
        
        return data_bytes, "application/octet-stream", filename
    
    def _export_hdf5(self, 
                    result: MonteCarloResult, 
                    config: ExportConfig) -> Tuple[bytes, str, str]:
        """Export result as HDF5 file."""
        
        hdf5_buffer = io.BytesIO()
        
        # Create datasets
        summary_df = self._create_summary_dataframe(result, config)
        
        with pd.HDFStore(hdf5_buffer, mode='w') as store:
            store.put('summary', summary_df, format='table')
            
            if config.include_metadata:
                config_df = self._create_config_dataframe(result.config)
                store.put('configuration', config_df, format='table')
            
            if config.include_statistics and result.simulation_summary:
                stats_df = self._create_statistics_dataframe(result.simulation_summary, config)
                store.put('statistics', stats_df, format='table')
            
            if config.include_percentiles and result.percentile_data:
                percentile_df = self._create_percentile_dataframe(result.percentile_data, config)
                store.put('percentiles', percentile_df, format='table')
        
        data_bytes = hdf5_buffer.getvalue()
        filename = f"monte_carlo_result_{self._get_timestamp()}.h5"
        
        return data_bytes, "application/octet-stream", filename
    
    def _prepare_export_data(self, 
                           result: MonteCarloResult, 
                           config: ExportConfig) -> Dict[str, Any]:
        """Prepare data structure for export."""
        
        export_data = {
            "export_metadata": {
                "export_timestamp": datetime.utcnow().isoformat(),
                "export_format": config.format,
                "exporter_version": "1.0.0"
            }
        }
        
        # Add simulation results
        export_data["results"] = {
            "expected_value": round(result.expected_value, config.decimal_places),
            "probability_of_loss": round(result.probability_of_loss, config.decimal_places),
            "value_at_risk_95": round(result.value_at_risk_95, config.decimal_places),
            "conditional_var_95": round(result.conditional_var_95, config.decimal_places)
        }
        
        # Add configuration
        if config.include_metadata:
            export_data["configuration"] = self._serialize_config(result.config, config)
        
        # Add statistics
        if config.include_statistics and result.simulation_summary:
            export_data["statistics"] = self._serialize_statistics(result.simulation_summary, config)
        
        # Add percentile data
        if config.include_percentiles and result.percentile_data:
            export_data["percentile_data"] = self._serialize_percentiles(result.percentile_data, config)
        
        return export_data
    
    def _create_summary_dataframe(self, 
                                result: MonteCarloResult, 
                                config: ExportConfig) -> pd.DataFrame:
        """Create summary DataFrame."""
        
        summary_data = {
            'Metric': [
                'Expected Value',
                'Probability of Loss',
                'Value at Risk (95%)',
                'Conditional VaR (95%)',
                'Initial Value',
                'Time Horizon (Years)',
                'Number of Simulations'
            ],
            'Value': [
                round(result.expected_value, config.decimal_places),
                round(result.probability_of_loss, config.decimal_places),
                round(result.value_at_risk_95, config.decimal_places),
                round(result.conditional_var_95, config.decimal_places),
                round(result.config.initial_value, config.decimal_places),
                result.config.time_horizon_years,
                result.config.num_simulations
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    def _create_config_dataframe(self, config: MonteCarloConfig) -> pd.DataFrame:
        """Create configuration DataFrame."""
        
        config_data = {
            'Parameter': [],
            'Value': []
        }
        
        # Add scalar parameters
        scalar_params = {
            'time_horizon_years': config.time_horizon_years,
            'num_simulations': config.num_simulations,
            'initial_value': config.initial_value
        }
        
        for param, value in scalar_params.items():
            config_data['Parameter'].append(param)
            config_data['Value'].append(value)
        
        # Add list parameters
        for i, ticker in enumerate(config.portfolio_tickers):
            config_data['Parameter'].append(f'ticker_{i+1}')
            config_data['Value'].append(ticker)
            
            if i < len(config.portfolio_weights):
                config_data['Parameter'].append(f'weight_{i+1}')
                config_data['Value'].append(config.portfolio_weights[i])
        
        return pd.DataFrame(config_data)
    
    def _create_statistics_dataframe(self, 
                                   statistics: Dict[str, Any], 
                                   config: ExportConfig) -> pd.DataFrame:
        """Create statistics DataFrame."""
        
        stats_data = {
            'Statistic': [],
            'Value': []
        }
        
        for key, value in statistics.items():
            if isinstance(value, (int, float)):
                stats_data['Statistic'].append(key.replace('_', ' ').title())
                stats_data['Value'].append(round(value, config.decimal_places))
        
        return pd.DataFrame(stats_data)
    
    def _create_percentile_dataframe(self, 
                                   percentile_data: Dict[str, Any], 
                                   config: ExportConfig) -> pd.DataFrame:
        """Create percentile data DataFrame."""
        
        # Extract time points (assuming they're consistent across percentiles)
        time_points = None
        percentile_columns = {}
        
        for percentile_key, data in percentile_data.items():
            if isinstance(data, dict) and 'values' in data:
                values = data['values']
                if isinstance(values, list):
                    percentile_columns[percentile_key] = [
                        round(v, config.decimal_places) for v in values
                    ]
                    
                    if time_points is None and 'time_points' in data:
                        time_points = data['time_points']
        
        # Create DataFrame
        df_data = {}
        
        if time_points:
            df_data['time_point'] = time_points
        
        df_data.update(percentile_columns)
        
        return pd.DataFrame(df_data)
    
    def _create_risk_metrics_dataframe(self, 
                                     result: MonteCarloResult, 
                                     config: ExportConfig) -> pd.DataFrame:
        """Create risk metrics DataFrame."""
        
        risk_data = {
            'Risk Metric': [
                'Value at Risk (95%)',
                'Conditional VaR (95%)',
                'Probability of Loss',
                'Expected Shortfall',
                'Downside Risk'
            ],
            'Value': [
                round(result.value_at_risk_95, config.decimal_places),
                round(result.conditional_var_95, config.decimal_places),
                round(result.probability_of_loss, config.decimal_places),
                round(result.conditional_var_95, config.decimal_places),  # Same as CVaR
                round(result.probability_of_loss * result.config.initial_value, config.decimal_places)
            ],
            'Description': [
                'Maximum expected loss at 95% confidence level',
                'Expected loss given that VaR threshold is exceeded',
                'Probability that portfolio value will be below initial value',
                'Average loss in worst 5% of scenarios',
                'Expected loss from downside scenarios'
            ]
        }
        
        return pd.DataFrame(risk_data)
    
    def _create_comprehensive_dataframe(self, 
                                      result: MonteCarloResult, 
                                      config: ExportConfig) -> pd.DataFrame:
        """Create comprehensive DataFrame for formats like Parquet."""
        
        # Start with basic results
        data = {
            'expected_value': [result.expected_value],
            'probability_of_loss': [result.probability_of_loss],
            'value_at_risk_95': [result.value_at_risk_95],
            'conditional_var_95': [result.conditional_var_95],
            'initial_value': [result.config.initial_value],
            'time_horizon_years': [result.config.time_horizon_years],
            'num_simulations': [result.config.num_simulations]
        }
        
        # Add portfolio composition
        for i, ticker in enumerate(result.config.portfolio_tickers):
            data[f'ticker_{i+1}'] = [ticker]
            if i < len(result.config.portfolio_weights):
                data[f'weight_{i+1}'] = [result.config.portfolio_weights[i]]
        
        # Add statistics if available
        if config.include_statistics and result.simulation_summary:
            for key, value in result.simulation_summary.items():
                if isinstance(value, (int, float)):
                    data[f'stat_{key}'] = [value]
        
        return pd.DataFrame(data)
    
    def _serialize_config(self, 
                         config: MonteCarloConfig, 
                         export_config: ExportConfig) -> Dict[str, Any]:
        """Serialize configuration for export."""
        
        config_dict = asdict(config)
        
        # Round numeric values
        for key, value in config_dict.items():
            if isinstance(value, float):
                config_dict[key] = round(value, export_config.decimal_places)
            elif isinstance(value, list) and value and isinstance(value[0], float):
                config_dict[key] = [round(v, export_config.decimal_places) for v in value]
        
        return config_dict
    
    def _serialize_statistics(self, 
                            statistics: Dict[str, Any], 
                            config: ExportConfig) -> Dict[str, Any]:
        """Serialize statistics for export."""
        
        serialized = {}
        
        for key, value in statistics.items():
            if isinstance(value, (int, float)):
                serialized[key] = round(value, config.decimal_places)
            else:
                serialized[key] = value
        
        return serialized
    
    def _serialize_percentiles(self, 
                             percentile_data: Dict[str, Any], 
                             config: ExportConfig) -> Dict[str, Any]:
        """Serialize percentile data for export."""
        
        serialized = {}
        
        for percentile_key, data in percentile_data.items():
            if isinstance(data, dict):
                serialized_data = {}
                
                for key, value in data.items():
                    if key == 'values' and isinstance(value, list):
                        serialized_data[key] = [
                            round(v, config.decimal_places) if isinstance(v, (int, float)) else v
                            for v in value
                        ]
                    else:
                        serialized_data[key] = value
                
                serialized[percentile_key] = serialized_data
            else:
                serialized[percentile_key] = data
        
        return serialized
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for special types."""
        
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return str(obj)
    
    def _get_timestamp(self) -> str:
        """Get timestamp string for filenames."""
        return datetime.now().strftime('%Y%m%d_%H%M%S')


class BatchExporter:
    """Batch exporter for multiple Monte Carlo results."""
    
    def __init__(self):
        """Initialize batch exporter."""
        self.exporter = ResultExporter()
    
    def export_multiple_results(self, 
                              results: List[MonteCarloResult], 
                              config: ExportConfig,
                              result_names: Optional[List[str]] = None) -> Tuple[bytes, str, str]:
        """
        Export multiple Monte Carlo results in a single file.
        
        Args:
            results: List of Monte Carlo results
            config: Export configuration
            result_names: Optional names for each result
            
        Returns:
            Tuple of (data_bytes, content_type, filename)
        """
        
        if not results:
            raise ExportError("No results provided for export")
        
        if result_names and len(result_names) != len(results):
            raise ExportError("Number of result names must match number of results")
        
        if not result_names:
            result_names = [f"result_{i+1}" for i in range(len(results))]
        
        if config.format == 'excel':
            return self._export_multiple_excel(results, result_names, config)
        elif config.format == 'json':
            return self._export_multiple_json(results, result_names, config)
        else:
            raise ExportError(f"Batch export not supported for format: {config.format}")
    
    def _export_multiple_excel(self, 
                             results: List[MonteCarloResult], 
                             result_names: List[str], 
                             config: ExportConfig) -> Tuple[bytes, str, str]:
        """Export multiple results as Excel workbook with multiple sheets."""
        
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            
            # Create comparison summary sheet
            comparison_df = self._create_comparison_dataframe(results, result_names, config)
            comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
            
            # Create individual sheets for each result
            for i, (result, name) in enumerate(zip(results, result_names)):
                sheet_name = name[:31]  # Excel sheet name limit
                
                summary_df = self.exporter._create_summary_dataframe(result, config)
                summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        data_bytes = excel_buffer.getvalue()
        filename = f"monte_carlo_comparison_{self.exporter._get_timestamp()}.xlsx"
        
        return data_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename
    
    def _export_multiple_json(self, 
                            results: List[MonteCarloResult], 
                            result_names: List[str], 
                            config: ExportConfig) -> Tuple[bytes, str, str]:
        """Export multiple results as JSON."""
        
        export_data = {
            "export_metadata": {
                "export_timestamp": datetime.utcnow().isoformat(),
                "export_format": config.format,
                "num_results": len(results),
                "exporter_version": "1.0.0"
            },
            "results": {}
        }
        
        for result, name in zip(results, result_names):
            result_data = self.exporter._prepare_export_data(result, config)
            export_data["results"][name] = result_data
        
        json_str = json.dumps(
            export_data, 
            indent=2,
            default=self.exporter._json_serializer
        )
        
        data_bytes = json_str.encode('utf-8')
        filename = f"monte_carlo_comparison_{self.exporter._get_timestamp()}.json"
        
        return data_bytes, "application/json", filename
    
    def _create_comparison_dataframe(self, 
                                   results: List[MonteCarloResult], 
                                   result_names: List[str], 
                                   config: ExportConfig) -> pd.DataFrame:
        """Create comparison DataFrame for multiple results."""
        
        comparison_data = {
            'Result Name': result_names,
            'Expected Value': [round(r.expected_value, config.decimal_places) for r in results],
            'Probability of Loss': [round(r.probability_of_loss, config.decimal_places) for r in results],
            'VaR 95%': [round(r.value_at_risk_95, config.decimal_places) for r in results],
            'CVaR 95%': [round(r.conditional_var_95, config.decimal_places) for r in results],
            'Time Horizon': [r.config.time_horizon_years for r in results],
            'Simulations': [r.config.num_simulations for r in results]
        }
        
        return pd.DataFrame(comparison_data)

cl
ass WebhookNotifier:
    """Webhook notification system for analytics results."""
    
    def __init__(self):
        """Initialize webhook notifier."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Portfolio-Analytics-Webhook/2.0.0'
        })
    
    async def send_notification(self, 
                              webhook_config: WebhookConfig,
                              event_type: str,
                              data: Dict[str, Any],
                              result_id: Optional[str] = None) -> bool:
        """
        Send webhook notification.
        
        Args:
            webhook_config: Webhook configuration
            event_type: Type of event (e.g., 'export_completed', 'analysis_finished')
            data: Event data to send
            result_id: Optional result identifier
            
        Returns:
            True if notification sent successfully, False otherwise
        """
        
        payload = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        
        if result_id:
            payload['result_id'] = result_id
        
        # Add webhook metadata
        payload['webhook_metadata'] = {
            'version': '2.0.0',
            'retry_attempt': 0
        }
        
        for attempt in range(webhook_config.retry_attempts):
            try:
                logger.info(f"Sending webhook notification (attempt {attempt + 1})")
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.request(
                        method=webhook_config.method,
                        url=webhook_config.url,
                        json=payload,
                        headers=webhook_config.headers,
                        timeout=webhook_config.timeout,
                        verify=webhook_config.verify_ssl
                    )
                )
                
                response.raise_for_status()
                
                logger.info(f"Webhook notification sent successfully to {webhook_config.url}")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Webhook notification attempt {attempt + 1} failed: {e}")
                
                if attempt < webhook_config.retry_attempts - 1:
                    await asyncio.sleep(webhook_config.retry_delay * (2 ** attempt))  # Exponential backoff
                    payload['webhook_metadata']['retry_attempt'] = attempt + 1
                else:
                    logger.error(f"All webhook notification attempts failed for {webhook_config.url}")
                    return False
            
            except Exception as e:
                logger.error(f"Unexpected error sending webhook notification: {e}")
                return False
        
        return False
    
    async def notify_export_completed(self,
                                    webhook_config: WebhookConfig,
                                    export_info: Dict[str, Any]) -> bool:
        """
        Send notification when export is completed.
        
        Args:
            webhook_config: Webhook configuration
            export_info: Export information
            
        Returns:
            True if notification sent successfully
        """
        
        return await self.send_notification(
            webhook_config=webhook_config,
            event_type='export_completed',
            data={
                'export_format': export_info.get('format'),
                'result_type': export_info.get('result_type'),
                'file_size_bytes': export_info.get('file_size'),
                'export_duration_seconds': export_info.get('duration'),
                'download_url': export_info.get('download_url'),
                'expires_at': export_info.get('expires_at')
            },
            result_id=export_info.get('result_id')
        )
    
    async def notify_analysis_completed(self,
                                      webhook_config: WebhookConfig,
                                      analysis_info: Dict[str, Any]) -> bool:
        """
        Send notification when analysis is completed.
        
        Args:
            webhook_config: Webhook configuration
            analysis_info: Analysis information
            
        Returns:
            True if notification sent successfully
        """
        
        return await self.send_notification(
            webhook_config=webhook_config,
            event_type='analysis_completed',
            data={
                'analysis_type': analysis_info.get('type'),
                'portfolio_id': analysis_info.get('portfolio_id'),
                'duration_seconds': analysis_info.get('duration'),
                'key_metrics': analysis_info.get('key_metrics', {}),
                'status': analysis_info.get('status', 'completed')
            },
            result_id=analysis_info.get('result_id')
        )
    
    async def notify_bulk_operation_completed(self,
                                            webhook_config: WebhookConfig,
                                            bulk_info: Dict[str, Any]) -> bool:
        """
        Send notification when bulk operation is completed.
        
        Args:
            webhook_config: Webhook configuration
            bulk_info: Bulk operation information
            
        Returns:
            True if notification sent successfully
        """
        
        return await self.send_notification(
            webhook_config=webhook_config,
            event_type='bulk_operation_completed',
            data={
                'batch_id': bulk_info.get('batch_id'),
                'total_operations': bulk_info.get('total_operations'),
                'completed': bulk_info.get('completed'),
                'failed': bulk_info.get('failed'),
                'duration_seconds': bulk_info.get('duration'),
                'operations': bulk_info.get('operations', [])
            }
        )


class BulkAnalyticsProcessor:
    """Processor for bulk analytics operations across multiple portfolios."""
    
    def __init__(self, analytics_service, exporter: AnalyticsExporter, webhook_notifier: WebhookNotifier):
        """
        Initialize bulk processor.
        
        Args:
            analytics_service: Analytics service instance
            exporter: Analytics exporter instance
            webhook_notifier: Webhook notifier instance
        """
        self.analytics_service = analytics_service
        self.exporter = exporter
        self.webhook_notifier = webhook_notifier
    
    async def process_bulk_operations(self,
                                    portfolios: List[Dict[str, Any]],
                                    operations: List[str],
                                    export_config: Optional[ExportConfig] = None,
                                    webhook_config: Optional[WebhookConfig] = None) -> Dict[str, Any]:
        """
        Process bulk analytics operations.
        
        Args:
            portfolios: List of portfolio configurations
            operations: List of operations to perform
            export_config: Optional export configuration
            webhook_config: Optional webhook configuration
            
        Returns:
            Bulk operation results
        """
        
        start_time = datetime.utcnow()
        batch_id = f"bulk_{start_time.strftime('%Y%m%d_%H%M%S')}_{id(self)}"
        
        logger.info(f"Starting bulk operations [{batch_id}]", extra={
            'portfolio_count': len(portfolios),
            'operations': operations
        })
        
        results = []
        completed = 0
        failed = 0
        
        try:
            # Process each portfolio
            for portfolio_idx, portfolio_data in enumerate(portfolios):
                portfolio_id = portfolio_data.get('id', f"portfolio_{portfolio_idx}")
                tickers = portfolio_data.get('tickers', [])
                weights = portfolio_data.get('weights', [])
                
                portfolio_results = {}
                
                # Process each operation for this portfolio
                for operation in operations:
                    try:
                        logger.info(f"Processing {operation} for portfolio {portfolio_id}")
                        
                        result = await self._execute_operation(
                            operation, portfolio_id, tickers, weights, portfolio_data
                        )
                        
                        portfolio_results[operation] = {
                            'status': 'completed',
                            'result': result,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                        
                        completed += 1
                        
                        # Send individual operation webhook if configured
                        if webhook_config:
                            await self.webhook_notifier.notify_analysis_completed(
                                webhook_config,
                                {
                                    'type': operation,
                                    'portfolio_id': portfolio_id,
                                    'result_id': getattr(result, 'id', None),
                                    'status': 'completed'
                                }
                            )
                        
                    except Exception as e:
                        logger.error(f"Operation {operation} failed for portfolio {portfolio_id}: {e}")
                        
                        portfolio_results[operation] = {
                            'status': 'failed',
                            'error': str(e),
                            'timestamp': datetime.utcnow().isoformat()
                        }
                        
                        failed += 1
                
                results.append({
                    'portfolio_id': portfolio_id,
                    'operations': portfolio_results
                })
                
                # Allow other tasks to run
                await asyncio.sleep(0)
            
            # Export results if configured
            export_info = None
            if export_config and results:
                try:
                    export_info = await self._export_bulk_results(results, export_config, batch_id)
                except Exception as e:
                    logger.error(f"Bulk export failed: {e}")
            
            # Calculate final statistics
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            bulk_result = {
                'batch_id': batch_id,
                'total_operations': len(portfolios) * len(operations),
                'completed': completed,
                'failed': failed,
                'duration_seconds': duration,
                'results': results,
                'export_info': export_info,
                'status': 'completed' if failed == 0 else ('partial' if completed > 0 else 'failed')
            }
            
            # Send bulk completion webhook if configured
            if webhook_config:
                await self.webhook_notifier.notify_bulk_operation_completed(
                    webhook_config,
                    bulk_result
                )
            
            logger.info(f"Bulk operations completed [{batch_id}]", extra={
                'completed': completed,
                'failed': failed,
                'duration_seconds': duration
            })
            
            return bulk_result
            
        except Exception as e:
            logger.error(f"Bulk operations failed [{batch_id}]: {e}")
            
            # Send failure webhook if configured
            if webhook_config:
                await self.webhook_notifier.send_notification(
                    webhook_config,
                    'bulk_operation_failed',
                    {
                        'batch_id': batch_id,
                        'error': str(e),
                        'completed': completed,
                        'failed': failed
                    }
                )
            
            raise AnalyticsError(f"Bulk operations failed: {str(e)}")
    
    async def _execute_operation(self,
                               operation: str,
                               portfolio_id: str,
                               tickers: List[str],
                               weights: List[float],
                               portfolio_data: Dict[str, Any]) -> Any:
        """Execute a single analytics operation."""
        
        if operation == 'backtest':
            from .models import BacktestConfig
            config = BacktestConfig(
                tickers=tickers,
                start_date=date.fromisoformat(portfolio_data.get('start_date', '2020-01-01')),
                end_date=date.fromisoformat(portfolio_data.get('end_date', date.today().isoformat())),
                strategy=portfolio_data.get('strategy', 'sharpe'),
                rebalance_frequency=portfolio_data.get('rebalance_frequency', 'monthly'),
                transaction_cost=portfolio_data.get('transaction_cost', 0.001),
                initial_capital=portfolio_data.get('initial_capital', 100000.0)
            )
            return self.analytics_service.run_backtest(config)
        
        elif operation == 'monte_carlo':
            from .models import MonteCarloConfig
            config = MonteCarloConfig(
                portfolio_tickers=tickers,
                portfolio_weights=weights,
                time_horizon_years=portfolio_data.get('time_horizon_years', 10),
                num_simulations=portfolio_data.get('num_simulations', 10000),
                confidence_levels=portfolio_data.get('confidence_levels', [0.05, 0.25, 0.5, 0.75, 0.95]),
                initial_value=portfolio_data.get('initial_value', 100000.0)
            )
            return self.analytics_service.run_monte_carlo(config)
        
        elif operation == 'risk_analysis':
            return self.analytics_service.analyze_risk(portfolio_id, tickers, weights)
        
        elif operation == 'performance':
            return self.analytics_service.track_performance(portfolio_id, tickers, weights)
        
        elif operation == 'dividends':
            return self.analytics_service.analyze_dividends(portfolio_id, tickers, weights)
        
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def _export_bulk_results(self,
                                 results: List[Dict[str, Any]],
                                 export_config: ExportConfig,
                                 batch_id: str) -> Dict[str, Any]:
        """Export bulk operation results."""
        
        try:
            # Flatten results for export
            flattened_results = []
            
            for portfolio_result in results:
                portfolio_id = portfolio_result['portfolio_id']
                
                for operation, operation_result in portfolio_result['operations'].items():
                    if operation_result['status'] == 'completed':
                        flattened_results.append(operation_result['result'])
            
            if not flattened_results:
                logger.warning("No successful results to export")
                return None
            
            # Export results
            data_bytes, content_type, filename = self.exporter.export_multiple_results(
                flattened_results, export_config
            )
            
            # In a real implementation, you would save the file and provide a download URL
            # For now, we'll just return the export information
            
            return {
                'filename': filename,
                'content_type': content_type,
                'file_size': len(data_bytes),
                'batch_id': batch_id,
                'export_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Bulk export failed: {e}")
            raise ExportError(f"Bulk export failed: {str(e)}")

    # Helper methods for enhanced export functionality
    
    def _get_result_type(self, result) -> str:
        """Get result type string."""
        return type(result).__name__.lower().replace('result', '').replace('analysis', '_analysis')
    
    def _apply_compression(self, data: bytes, filename: str, compression: Optional[CompressionType]) -> Tuple[bytes, str]:
        """Apply compression to data and update filename."""
        
        if not compression or compression == CompressionType.NONE:
            return data, filename
        
        if compression == CompressionType.GZIP:
            compressed_data = gzip.compress(data)
            return compressed_data, f"{filename}.gz"
        
        elif compression == CompressionType.ZIP:
            import zipfile
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(filename, data)
            return zip_buffer.getvalue(), f"{filename}.zip"
        
        elif compression == CompressionType.BZIP2:
            import bz2
            compressed_data = bz2.compress(data)
            return compressed_data, f"{filename}.bz2"
        
        else:
            logger.warning(f"Unsupported compression type: {compression}")
            return data, filename
    
    def _apply_pagination(self, results: List, result_names: List[str], 
                         pagination_config: PaginationConfig) -> Tuple[List, List[str]]:
        """Apply pagination to results."""
        
        start_idx = 0
        end_idx = pagination_config.page_size
        
        if pagination_config.max_pages:
            max_items = pagination_config.page_size * pagination_config.max_pages
            end_idx = min(end_idx, max_items)
        
        paginated_results = results[start_idx:end_idx]
        paginated_names = result_names[start_idx:end_idx]
        
        return paginated_results, paginated_names
    
    def _create_result_dataframe(self, result, config: ExportConfig) -> pd.DataFrame:
        """Create main results DataFrame based on result type."""
        
        if isinstance(result, MonteCarloResult):
            return self._create_monte_carlo_dataframe(result, config)
        elif isinstance(result, BacktestResult):
            return self._create_backtest_dataframe(result, config)
        elif isinstance(result, RiskAnalysis):
            return self._create_risk_analysis_dataframe(result, config)
        elif isinstance(result, PerformanceMetrics):
            return self._create_performance_dataframe(result, config)
        elif isinstance(result, DividendAnalysis):
            return self._create_dividend_analysis_dataframe(result, config)
        else:
            raise ExportError(f"Unsupported result type: {type(result)}")
    
    def _create_monte_carlo_dataframe(self, result: MonteCarloResult, config: ExportConfig) -> pd.DataFrame:
        """Create DataFrame for Monte Carlo results."""
        
        data = {
            'Metric': [
                'Expected Value',
                'Probability of Loss',
                'Value at Risk (95%)',
                'Conditional VaR (95%)',
                'Initial Value',
                'Time Horizon (Years)',
                'Number of Simulations'
            ],
            'Value': [
                round(result.expected_value, config.decimal_places),
                round(result.probability_of_loss, config.decimal_places),
                round(result.value_at_risk_95, config.decimal_places),
                round(result.conditional_var_95, config.decimal_places),
                round(result.config.initial_value, config.decimal_places),
                result.config.time_horizon_years,
                result.config.num_simulations
            ]
        }
        
        return pd.DataFrame(data)
    
    def _create_backtest_dataframe(self, result: BacktestResult, config: ExportConfig) -> pd.DataFrame:
        """Create DataFrame for backtest results."""
        
        data = {
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Volatility',
                'Sharpe Ratio',
                'Maximum Drawdown',
                'Calmar Ratio',
                'Transaction Costs',
                'Number of Rebalances',
                'Final Value'
            ],
            'Value': [
                round(result.total_return, config.decimal_places),
                round(result.annualized_return, config.decimal_places),
                round(result.volatility, config.decimal_places),
                round(result.sharpe_ratio, config.decimal_places),
                round(result.max_drawdown, config.decimal_places),
                round(result.calmar_ratio, config.decimal_places),
                round(result.transaction_costs, config.decimal_places),
                result.num_rebalances,
                round(result.final_value, config.decimal_places)
            ]
        }
        
        return pd.DataFrame(data)
    
    def _create_risk_analysis_dataframe(self, result: RiskAnalysis, config: ExportConfig) -> pd.DataFrame:
        """Create DataFrame for risk analysis results."""
        
        data = {
            'Metric': [
                'Portfolio Beta',
                'Tracking Error',
                'Information Ratio',
                'Value at Risk (95%)',
                'Conditional VaR (95%)',
                'Maximum Drawdown',
                'Concentration Risk'
            ],
            'Value': [
                round(result.portfolio_beta, config.decimal_places),
                round(result.tracking_error, config.decimal_places),
                round(result.information_ratio, config.decimal_places),
                round(result.var_95, config.decimal_places),
                round(result.cvar_95, config.decimal_places),
                round(result.max_drawdown, config.decimal_places),
                round(result.concentration_risk, config.decimal_places)
            ]
        }
        
        return pd.DataFrame(data)
    
    def _create_performance_dataframe(self, result: PerformanceMetrics, config: ExportConfig) -> pd.DataFrame:
        """Create DataFrame for performance metrics."""
        
        data = {
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Volatility',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Alpha',
                'Beta',
                'R-Squared',
                'Tracking Error',
                'Information Ratio'
            ],
            'Value': [
                round(result.total_return, config.decimal_places),
                round(result.annualized_return, config.decimal_places),
                round(result.volatility, config.decimal_places),
                round(result.sharpe_ratio, config.decimal_places),
                round(result.sortino_ratio, config.decimal_places),
                round(result.alpha, config.decimal_places),
                round(result.beta, config.decimal_places),
                round(result.r_squared, config.decimal_places),
                round(result.tracking_error, config.decimal_places),
                round(result.information_ratio, config.decimal_places)
            ]
        }
        
        return pd.DataFrame(data)
    
    def _create_dividend_analysis_dataframe(self, result: DividendAnalysis, config: ExportConfig) -> pd.DataFrame:
        """Create DataFrame for dividend analysis results."""
        
        data = {
            'Metric': [
                'Current Yield',
                'Projected Annual Income',
                'Dividend Growth Rate',
                'Payout Ratio',
                'Dividend Coverage',
                'Income Sustainability Score'
            ],
            'Value': [
                round(result.current_yield, config.decimal_places),
                round(result.projected_annual_income, config.decimal_places),
                round(result.dividend_growth_rate, config.decimal_places),
                round(result.payout_ratio, config.decimal_places),
                round(result.dividend_coverage, config.decimal_places),
                round(result.income_sustainability_score, config.decimal_places)
            ]
        }
        
        return pd.DataFrame(data)
    
    def _create_metadata_dataframe(self, result) -> pd.DataFrame:
        """Create metadata DataFrame for any result type."""
        
        metadata = {
            'Property': ['Result Type', 'Generated At'],
            'Value': [type(result).__name__, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        
        # Add result-specific metadata
        if hasattr(result, 'config'):
            if hasattr(result.config, 'tickers'):
                metadata['Property'].append('Tickers')
                metadata['Value'].append(', '.join(result.config.tickers))
            if hasattr(result.config, 'portfolio_tickers'):
                metadata['Property'].append('Portfolio Tickers')
                metadata['Value'].append(', '.join(result.config.portfolio_tickers))
        
        if hasattr(result, 'portfolio_id'):
            metadata['Property'].append('Portfolio ID')
            metadata['Value'].append(result.portfolio_id)
        
        if hasattr(result, 'analysis_date'):
            metadata['Property'].append('Analysis Date')
            metadata['Value'].append(result.analysis_date.isoformat())
        
        return pd.DataFrame(metadata)
    
    def _create_returns_dataframe(self, returns_data: Dict[str, Any], config: ExportConfig) -> pd.DataFrame:
        """Create DataFrame for returns data."""
        
        if 'returns' in returns_data and isinstance(returns_data['returns'], list):
            return pd.DataFrame({
                'Date': returns_data.get('dates', range(len(returns_data['returns']))),
                'Return': [round(r, config.decimal_places) for r in returns_data['returns']]
            })
        
        return pd.DataFrame({'Message': ['Returns data not available in expected format']})
    
    def _create_allocation_dataframe(self, allocation_data: Dict[str, Any], config: ExportConfig) -> pd.DataFrame:
        """Create DataFrame for allocation data."""
        
        if 'allocations' in allocation_data:
            allocations = allocation_data['allocations']
            if isinstance(allocations, dict):
                return pd.DataFrame([
                    {'Asset': asset, 'Allocation': round(weight, config.decimal_places)}
                    for asset, weight in allocations.items()
                ])
        
        return pd.DataFrame({'Message': ['Allocation data not available in expected format']})
    
    def _create_correlation_dataframe(self, correlation_data: Dict[str, Any], config: ExportConfig) -> pd.DataFrame:
        """Create DataFrame for correlation data."""
        
        if 'correlation_matrix' in correlation_data:
            corr_matrix = correlation_data['correlation_matrix']
            if isinstance(corr_matrix, dict):
                # Convert correlation matrix to DataFrame
                assets = list(corr_matrix.keys())
                data = []
                
                for asset1 in assets:
                    row = [asset1]
                    for asset2 in assets:
                        corr_value = corr_matrix.get(asset1, {}).get(asset2, 0)
                        row.append(round(corr_value, config.decimal_places))
                    data.append(row)
                
                columns = ['Asset'] + assets
                return pd.DataFrame(data, columns=columns)
        
        return pd.DataFrame({'Message': ['Correlation data not available in expected format']})
    
    def _create_factor_exposure_dataframe(self, factor_exposures: Dict[str, float], config: ExportConfig) -> pd.DataFrame:
        """Create DataFrame for factor exposures."""
        
        return pd.DataFrame([
            {'Factor': factor, 'Exposure': round(exposure, config.decimal_places)}
            for factor, exposure in factor_exposures.items()
        ])
    
    def _create_dividend_detail_dataframe(self, dividend_data: Dict[str, Any], config: ExportConfig) -> pd.DataFrame:
        """Create DataFrame for detailed dividend data."""
        
        if 'dividend_details' in dividend_data:
            details = dividend_data['dividend_details']
            if isinstance(details, list):
                return pd.DataFrame(details)
        
        return pd.DataFrame({'Message': ['Dividend detail data not available in expected format']})
    
    def _prepare_export_data(self, result, config: ExportConfig) -> Dict[str, Any]:
        """Prepare comprehensive export data structure."""
        
        export_data = {
            "export_metadata": {
                "export_timestamp": datetime.utcnow().isoformat(),
                "export_format": config.format.value,
                "result_type": self._get_result_type(result),
                "exporter_version": "2.0.0"
            }
        }
        
        # Add main result data
        export_data["result"] = self._prepare_result_data(result, config)
        
        # Add metadata if requested
        if config.include_metadata:
            export_data["metadata"] = self._prepare_metadata(result)
        
        return export_data
    
    def _prepare_result_data(self, result, config: ExportConfig) -> Dict[str, Any]:
        """Prepare result data for export."""
        
        # Convert result to dictionary, handling Pydantic models
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        else:
            result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else {}
        
        # Round numeric values
        for key, value in result_dict.items():
            if isinstance(value, float):
                result_dict[key] = round(value, config.decimal_places)
            elif isinstance(value, list) and value and isinstance(value[0], float):
                result_dict[key] = [round(v, config.decimal_places) for v in value]
        
        return result_dict
    
    def _prepare_metadata(self, result) -> Dict[str, Any]:
        """Prepare metadata for export."""
        
        metadata = {
            'result_type': type(result).__name__,
            'generated_at': datetime.now().isoformat()
        }
        
        # Add result-specific metadata
        if hasattr(result, 'portfolio_id'):
            metadata['portfolio_id'] = result.portfolio_id
        
        if hasattr(result, 'analysis_date'):
            metadata['analysis_date'] = result.analysis_date.isoformat()
        
        if hasattr(result, 'calculation_date'):
            metadata['calculation_date'] = result.calculation_date.isoformat()
        
        return metadata
    
    def _export_multiple_csv(self, results: List, result_names: List[str], config: ExportConfig) -> Tuple[bytes, str, str]:
        """Export multiple results as CSV with sections."""
        
        csv_buffer = io.StringIO()
        
        for i, (result, name) in enumerate(zip(results, result_names)):
            if i > 0:
                csv_buffer.write('\n\n')
            
            csv_buffer.write(f'# {name}\n')
            
            result_df = self._create_result_dataframe(result, config)
            result_df.to_csv(csv_buffer, index=False)
        
        data_bytes = csv_buffer.getvalue().encode('utf-8')
        
        # Apply compression if requested
        data_bytes, filename = self._apply_compression(
            data_bytes,
            f"multiple_results_{self._get_timestamp()}.csv",
            config.compression
        )
        
        return data_bytes, "text/csv", filename