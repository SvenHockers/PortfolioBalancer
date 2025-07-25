"""Export utilities for Monte Carlo simulation results."""

import json
import csv
import io
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
from dataclasses import dataclass, asdict

from .models import MonteCarloResult, MonteCarloConfig
from .exceptions import ExportError


@dataclass
class ExportConfig:
    """Configuration for data export."""
    
    format: str
    include_raw_data: bool = False
    include_statistics: bool = True
    include_percentiles: bool = True
    include_metadata: bool = True
    compression: Optional[str] = None
    decimal_places: int = 4
    
    def __post_init__(self):
        valid_formats = ['json', 'csv', 'excel', 'parquet', 'hdf5']
        if self.format.lower() not in valid_formats:
            raise ValueError(f"Format must be one of {valid_formats}")
        
        self.format = self.format.lower()


class ResultExporter:
    """Exporter for Monte Carlo simulation results in various formats."""
    
    def __init__(self):
        """Initialize result exporter."""
        self.supported_formats = ['json', 'csv', 'excel', 'parquet', 'hdf5']
    
    def export_result(self, 
                     result: MonteCarloResult, 
                     config: ExportConfig) -> Tuple[bytes, str, str]:
        """
        Export Monte Carlo result in specified format.
        
        Args:
            result: Monte Carlo simulation result
            config: Export configuration
            
        Returns:
            Tuple of (data_bytes, content_type, filename)
        """
        try:
            if config.format == 'json':
                return self._export_json(result, config)
            elif config.format == 'csv':
                return self._export_csv(result, config)
            elif config.format == 'excel':
                return self._export_excel(result, config)
            elif config.format == 'parquet':
                return self._export_parquet(result, config)
            elif config.format == 'hdf5':
                return self._export_hdf5(result, config)
            else:
                raise ExportError(f"Unsupported export format: {config.format}")
                
        except Exception as e:
            raise ExportError(f"Export failed: {str(e)}")
    
    def _export_json(self, 
                    result: MonteCarloResult, 
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
        if config.compression == 'gzip':
            import gzip
            data_bytes = gzip.compress(json_str.encode('utf-8'))
            filename = f"monte_carlo_result_{self._get_timestamp()}.json.gz"
        else:
            data_bytes = json_str.encode('utf-8')
            filename = f"monte_carlo_result_{self._get_timestamp()}.json"
        
        return data_bytes, "application/json", filename
    
    def _export_csv(self, 
                   result: MonteCarloResult, 
                   config: ExportConfig) -> Tuple[bytes, str, str]:
        """Export result as CSV."""
        
        # Create main results DataFrame
        results_data = {
            'metric': [],
            'value': []
        }
        
        # Add basic results
        results_data['metric'].extend([
            'expected_value',
            'probability_of_loss', 
            'value_at_risk_95',
            'conditional_var_95'
        ])
        
        results_data['value'].extend([
            round(result.expected_value, config.decimal_places),
            round(result.probability_of_loss, config.decimal_places),
            round(result.value_at_risk_95, config.decimal_places),
            round(result.conditional_var_95, config.decimal_places)
        ])
        
        # Add configuration data
        if config.include_metadata:
            config_data = asdict(result.config)
            for key, value in config_data.items():
                if isinstance(value, (int, float)):
                    results_data['metric'].append(f'config_{key}')
                    results_data['value'].append(value)
        
        # Add statistics if available
        if config.include_statistics and result.simulation_summary:
            for key, value in result.simulation_summary.items():
                if isinstance(value, (int, float)):
                    results_data['metric'].append(f'stat_{key}')
                    results_data['value'].append(round(value, config.decimal_places))
        
        # Create DataFrame and export to CSV
        df = pd.DataFrame(results_data)
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # Handle percentile data if requested
        if config.include_percentiles and result.percentile_data:
            csv_buffer.write('\n\n# Percentile Data\n')
            percentile_df = self._create_percentile_dataframe(result.percentile_data, config)
            percentile_df.to_csv(csv_buffer, index=False)
        
        data_bytes = csv_buffer.getvalue().encode('utf-8')
        filename = f"monte_carlo_result_{self._get_timestamp()}.csv"
        
        return data_bytes, "text/csv", filename
    
    def _export_excel(self, 
                     result: MonteCarloResult, 
                     config: ExportConfig) -> Tuple[bytes, str, str]:
        """Export result as Excel workbook."""
        
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            
            # Summary sheet
            summary_data = self._create_summary_dataframe(result, config)
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
            
            # Configuration sheet
            if config.include_metadata:
                config_df = self._create_config_dataframe(result.config)
                config_df.to_excel(writer, sheet_name='Configuration', index=False)
            
            # Statistics sheet
            if config.include_statistics and result.simulation_summary:
                stats_df = self._create_statistics_dataframe(result.simulation_summary, config)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Percentile data sheet
            if config.include_percentiles and result.percentile_data:
                percentile_df = self._create_percentile_dataframe(result.percentile_data, config)
                percentile_df.to_excel(writer, sheet_name='Percentiles', index=False)
            
            # Risk metrics sheet
            risk_df = self._create_risk_metrics_dataframe(result, config)
            risk_df.to_excel(writer, sheet_name='Risk_Metrics', index=False)
        
        data_bytes = excel_buffer.getvalue()
        filename = f"monte_carlo_result_{self._get_timestamp()}.xlsx"
        
        return data_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename
    
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