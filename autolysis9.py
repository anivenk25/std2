import os
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import traceback
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys
import shutil
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from functools import wraps
import matplotlib.colors as mcolors
from concurrent.futures import ThreadPoolExecutor
from matplotlib.figure import Figure
import logging
import coloredlogs

# Configure logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                   fmt='%(asctime)s - %(levelname)s - %(message)s')

def timeit(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
        return result
    return wrapper

@dataclass
class AnalysisConfig:
    """Configuration settings for analysis with comprehensive parameters"""
    max_retries: int = 3
    confidence_threshold: float = 0.95
    visualization_dpi: int = 300
    token_limit: int = 4000
    output_dir: Path = Path("output")
    time_limit: int = 180
    color_blind_friendly: bool = True
    batch_size: int = 1000
    parallel_processing: bool = True
    cache_results: bool = True
    max_threads: int = 4

class APIClient:
    """Enhanced API client with robust error handling and rate limiting"""
    def __init__(self, config: AnalysisConfig):
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise EnvironmentError("AIPROXY_TOKEN environment variable not set")
        
        self.config = config
        self.proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # seconds between requests

    def _enforce_rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _optimize_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Optimize prompts to reduce token usage"""
        optimized_messages = []
        for msg in messages:
            content = msg['content']
            if len(content) > self.config.token_limit:
                content = content[:self.config.token_limit] + "..."
            optimized_messages.append({"role": msg['role'], "content": content})
        return optimized_messages

    @timeit
    def make_request(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make API request with comprehensive error handling and retries"""
        messages = self._optimize_prompt(messages)
        
        for attempt in range(self.config.max_retries):
            try:
                self._enforce_rate_limit()
                
                data = {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": self.config.token_limit
                }
                
                response = requests.post(
                    self.proxy_url,
                    headers=self.headers,
                    json=data,
                    timeout=30
                )
                response.raise_for_status()
                
                return response.json()['choices'][0]['message']['content']
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    logger.error("All API request attempts failed")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
    
        return None

class StatisticalMethods:
    """Comprehensive statistical analysis methods with error handling"""
    
    @staticmethod
    def basic_stats(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistical measures with enhanced error checking"""
        try:
            stats_dict = {
                'summary': data.describe(),
                'missing': data.isnull().sum(),
                'dtypes': data.dtypes,
                'skewness': data.select_dtypes(include=[np.number]).skew(),
                'kurtosis': data.select_dtypes(include=[np.number]).kurtosis(),
                'variance': data.select_dtypes(include=[np.number]).var()
            }
            
            # Add correlation analysis for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                stats_dict['correlation'] = data[numeric_cols].corr()
                
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error in basic_stats: {str(e)}")
            return {'error': str(e)}

    @staticmethod
    def normality_test(data: pd.Series) -> Dict[str, Any]:
        """Enhanced normality testing with multiple methods"""
        try:
            if len(data) < 3:
                return {'error': 'Insufficient data for normality test'}
                
            clean_data = data.dropna()
            
            # Multiple normality tests
            shapiro_stat, shapiro_p = stats.shapiro(clean_data)
            ks_stat, ks_p = stats.kstest(clean_data, 'norm')
            normal_stat, normal_p = stats.normaltest(clean_data)
            
            results = {
                'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p},
                'normal_test': {'statistic': normal_stat, 'p_value': normal_p},
                'skewness': stats.skew(clean_data),
                'kurtosis': stats.kurtosis(clean_data),
                'is_normal': all(p > 0.05 for test in ['shapiro', 'kolmogorov_smirnov', 'normal_test']
                                for p in [locals()[f"{test}_p"]])
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in normality_test: {str(e)}")
            return {'error': str(e)}

    @staticmethod
    def outlier_detection(data: pd.Series) -> Dict[str, Any]:
        """Comprehensive outlier detection using multiple methods"""
        try:
            clean_data = data.dropna()
            
            # IQR method
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = clean_data[(clean_data < (Q1 - 1.5 * IQR)) | 
                                    (clean_data > (Q3 + 1.5 * IQR))]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(clean_data))
            z_score_outliers = clean_data[z_scores > 3]
            
            # Modified Z-score method
            median = np.median(clean_data)
            mad = stats.median_abs_deviation(clean_data)
            modified_z_scores = 0.6745 * (clean_data - median) / mad
            modified_z_outliers = clean_data[np.abs(modified_z_scores) > 3.5]
            
            results = {
                'iqr_method': {
                    'outlier_count': len(iqr_outliers),
                    'outlier_percentage': (len(iqr_outliers) / len(clean_data)) * 100,
                    'outlier_values': iqr_outliers.tolist()
                },
                'z_score_method': {
                    'outlier_count': len(z_score_outliers),
                    'outlier_percentage': (len(z_score_outliers) / len(clean_data)) * 100,
                    'outlier_values': z_score_outliers.tolist()
                },
                'modified_z_score_method': {
                    'outlier_count': len(modified_z_outliers),
                    'outlier_percentage': (len(modified_z_outliers) / len(clean_data)) * 100,
                    'outlier_values': modified_z_outliers.tolist()
                },
                'consensus_outliers': {
                    'values': list(set(iqr_outliers) & set(z_score_outliers) & 
                                 set(modified_z_outliers))
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in outlier_detection: {str(e)}")
            return {'error': str(e)}

    @staticmethod
    def dimension_reduction(data: pd.DataFrame, n_components: int = 2) -> Dict[str, Any]:
        """Enhanced dimension reduction with multiple techniques"""
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)
            
            # Calculate feature importance
            feature_importance = np.abs(pca.components_)
            feature_importance = feature_importance / np.sum(feature_importance, axis=1)[:, np.newaxis]
            
            results = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'components': pca_result.tolist(),
                'feature_importance': {
                    col: importance.tolist()
                    for col, importance in zip(data.columns, feature_importance.T)
                },
                'n_components_95var': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in dimension_reduction: {str(e)}")
            return {'error': str(e)}

class VisualizationStrategy(ABC):
    """Abstract base class for visualization strategies with accessibility support"""
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.color_schemes = {
            'default': sns.color_palette("husl", 8),
            'colorblind': sns.color_palette("colorblind", 8)
        }
        
    @abstractmethod
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        pass
        
    def _setup_plot_style(self) -> None:
        """Configure plot style with accessibility considerations"""
        plt.style.use('seaborn')
        if self.config.color_blind_friendly:
            sns.set_palette(self.color_schemes['colorblind'])
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

class CorrelationHeatmap(VisualizationStrategy):
    """Generate correlation heatmap with enhanced features"""
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        try:
            self._setup_plot_style()
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                logger.warning("No numeric columns available for correlation heatmap")
                return
            
            plt.figure(figsize=(12, 8))
            correlation_matrix = numeric_df.corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(correlation_matrix), k=1)
            
            # Generate heatmap with improved accessibility
            sns.heatmap(correlation_matrix,
                       annot=True,
                       cmap='RdYlBu_r',
                       center=0,
                       mask=mask,
                       fmt='.2f',
                       square=True,
                       cbar_kws={'label': 'Correlation Coefficient'})
            
            plt.title(f"Correlation Heatmap - {title}")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=self.config.visualization_dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")

class DistributionPlot(VisualizationStrategy):
    """Generate distribution plots with enhanced features"""
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        try:
            self._setup_plot_style()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            n_cols = len(numeric_cols)
            
            if n_cols == 0:
                logger.warning("No numeric columns available for distribution plots")
                return
            
            fig = plt.figure(figsize=(15, 5 * ((n_cols + 1) // 2)))
            
            for i, col in enumerate(numeric_cols, 1):
                ax = plt.subplot(((n_cols + 1) // 2), 2, i)
                
                # Create histogram with KDE
                sns.histplot(data=df[col], kde=True, ax=ax)
                
                # Add descriptive statistics
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()
                
                ax.axvline(mean_val, color='r', linestyle='--', alpha=0.8)
                ax.axvline(median_val, color='g', linestyle='--', alpha=0.8)
                
                # Add legend with statistics
                ax.text(0.95, 0.95,
                       f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}',
                       transform=ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.suptitle(f"Distribution Analysis - {title}")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=self.config.visualization_dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating distribution plots: {str(e)}")

class ScatterMatrix(VisualizationStrategy):
    """Generate scatter matrix for multivariate analysis"""
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        try:
            self._setup_plot_style()
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                logger.warning("Insufficient numeric columns for scatter matrix")
                return
            
            # Select only the first 5 numeric columns to avoid overcrowding
            plot_df = numeric_df.iloc[:, :5]
            
            fig = sns.pairplot(plot_df, diag_kind='kde', plot_kws={'alpha': 0.6})
            fig.fig.suptitle(f"Scatter Matrix - {title}", y=1.02)
            
            # Save with high resolution
            fig.savefig(fig_path, dpi=self.config.visualization_dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating scatter matrix: {str(e)}")

class TimeSeriesPlot(VisualizationStrategy):
    """Generate time series visualization if applicable"""
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        try:
            self._setup_plot_style()
            
            # Check for datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if not datetime_cols.empty:
                time_col = datetime_cols[0]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                plt.figure(figsize=(15, 8))
                for col in numeric_cols[:3]:  # Plot first 3 numeric columns
                    plt.plot(df[time_col], df[col], label=col, alpha=0.7)
                
                plt.title(f"Time Series Analysis - {title}")
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plt.savefig(fig_path, dpi=self.config.visualization_dpi, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {str(e)}")

class VisionAnalyzer:
    """Handle visual analysis of data distributions and patterns"""
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.api_client = APIClient(config)
        
    def analyze_visualization(self, fig_path: Path) -> str:
        """Analyze visualization and generate insights"""
        try:
            # Convert plot to base64 for API
            with open(fig_path, 'rb') as f:
                image_data = f.read()
            
            prompt = f"""
            Analyze this data visualization and provide insights on:
            1. Key patterns and trends
            2. Distribution characteristics
            3. Potential outliers or anomalies
            4. Relationships between variables
            5. Recommendations for further analysis
            
            Focus on actionable insights that would be valuable for decision-making.
            """
            
            messages = [
                {"role": "system", "content": "You are a data visualization expert."},
                {"role": "user", "content": prompt}
            ]
            
            return self.api_client.make_request(messages) or "Visual analysis unavailable"
            
        except Exception as e:
            logger.error(f"Error in visualization analysis: {str(e)}")
            return "Error analyzing visualization"

class DataAnalyzer:
    """Enhanced data analyzer with comprehensive analysis capabilities"""
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.api_client = APIClient(config)
        self.stats_analyzer = StatisticalAnalyzer()
        self.vision_analyzer = VisionAnalyzer(config)
        self.visualization_strategies = [
            CorrelationHeatmap(config),
            DistributionPlot(config),
            ScatterMatrix(config),
            TimeSeriesPlot(config)
        ]
        self.plots: List[str] = []
        
    @timeit
    def analyze_dataset(self, file_path: str) -> Dict[str, Any]:
        """Main method to analyze the dataset with comprehensive error handling"""
        try:
            start_time = time.time()
            results = {}
            
            self._create_output_directory()
            df = self._load_and_validate_dataset(file_path)
            logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            
            # Perform analysis in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
                futures = {
                    'stats': executor.submit(self._analyze_statistics, df),
                    'visualizations': executor.submit(self._generate_visualizations, df),
                    'narrative': executor.submit(self._generate_narrative, df)
                }
                
                # Collect results
                results['stats'] = futures['stats'].result()
                self.plots = futures['visualizations'].result()
                narrative = futures['narrative'].result()
                
            # Generate combined insights
            results['insights'] = self._generate_insights(df, results['stats'])
            
            # Generate final report
            if time.time() - start_time < self.config.time_limit:
                self._generate_report(results, narrative)
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            traceback.print_exc()
            return {'error': str(e)}

    def _create_output_directory(self) -> None:
        """Create or clean output directory"""
        if self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True)

    def _load_and_validate_dataset(self, file_path: str) -> pd.DataFrame:
        """Load and validate dataset with enhanced error handling"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        if path.suffix.lower() not in ['.csv', '.xlsx', '.parquet']:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        try:
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            elif path.suffix.lower() == '.xlsx':
                df = pd.read_excel(file_path)
            else:  # parquet
                df = pd.read_parquet(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Convert date columns to datetime
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    continue
        
        return df

    def _analyze_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis with parallel processing"""
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
                futures = {
                    'basic_stats': executor.submit(StatisticalMethods.basic_stats, df),
                    'normality_tests': executor.submit(self._analyze_normality, df),
                    'outliers': executor.submit(self._analyze_outliers, df),
                    'dimension_reduction': executor.submit(self._analyze_dimensions, df)
                }
                
                return {k: v.result() for k, v in futures.items()}
                
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")
            return {'error': str(e)}

    def _analyze_normality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze normality for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        return {
            col: StatisticalMethods.normality_test(numeric_df[col])
            for col in numeric_df.columns
        }

    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        return {
            col: StatisticalMethods.outlier_detection(numeric_df[col])
            for col in numeric_df.columns
        }

    def _analyze_dimensions(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Perform dimension reduction analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 2:
            return StatisticalMethods.dimension_reduction(numeric_df)
        return None

    @timeit
    def _generate_visualizations(self, df: pd.DataFrame) -> List[str]:
        """Generate visualizations with parallel processing"""
        plots = []
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
                futures = []
                for i, strategy in enumerate(self.visualization_strategies):
                    viz_path = self.config.output_dir / f'visualization_{i}.png'
                    future = executor.submit(
                        strategy.create_visualization,
                        df,
                        viz_path,
                        f"Analysis {i+1}"
                    )
                    futures.append((viz_path.name, future))
                
                # Collect results
                for plot_name, future in futures:
                    try:
                        future.result()  # Wait for completion
                        plots.append(plot_name)
                    except Exception as e:
                        logger.error(f"Error generating {plot_name}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in visualization generation: {str(e)}")
            
        return plots

    def _generate_insights(self, df: pd.DataFrame, stats: Dict[str, Any]) -> str:
        """Generate insights with enhanced context awareness"""
        try:
            context = self._determine_context(df)
            
            prompt = f"""
            Analyze this {context} dataset with the following information:

            1. Dataset Statistics:
            {json.dumps(stats.get('basic_stats', {}), default=str)}

            2. Advanced Analysis:
            {json.dumps(stats, default=str)}

            Provide detailed insights on:
            1. Key patterns and trends
            2. Statistical significance of findings
            3. Relationships between variables
            4. Anomalies and outliers
            5. Business implications
            6. Actionable recommendations

            Format with clear sections and evidence-based conclusions.
            """

            messages = [
                {"role": "system", "content": "You are a senior data scientist specializing in statistical analysis."},
                {"role": "user", "content": prompt}
            ]

            insights = self.api_client.make_request(messages)
            return insights if insights else "Insight generation unavailable"
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return "Error generating insights"

    def _determine_context(self, df: pd.DataFrame) -> str:
        """Determine dataset context using column names and values"""
        columns_str = ' '.join(df.columns.str.lower())
        
        contexts = {
            'financial': ['price', 'cost', 'revenue', 'profit', 'sales'],
            'customer': ['customer', 'client', 'user', 'satisfaction'],
            'product': ['product', 'item', 'sku', 'inventory'],
            'temporal': ['date', 'time', 'year', 'month'],
            'geographic': ['country', 'region', 'city', 'location']
        }
        
        matched_contexts = []
        for context, keywords in contexts.items():
            if any(keyword in columns_str for keyword in keywords):
                matched_contexts.append(context)
                
        return ' and '.join(matched_contexts) if matched_contexts else "general"

    @timeit
    def _generate_narrative(self, df: pd.DataFrame) -> str:
        """Generate narrative with enhanced storytelling"""
        try:
            context = self._determine_context(df)
            sample_data = df.head(5).to_dict()
            
            prompt = f"""
            Create an engaging data story for this {context} dataset.
            
            Sample data:
            {json.dumps(sample_data, default=str)}
            
            Requirements:
            1. Start with an executive summary
            2. Highlight key findings and their significance
            3. Include specific examples and numbers
            4. Discuss implications and recommendations
            5. Use clear sections and professional tone
            6. End with actionable next steps
            
            Focus on business value and decision-making impact.
            """
            
            messages = [
                {"role": "system", "content": "You are a data storyteller who transforms analysis into compelling narratives."},
                {"role": "user", "content": prompt}
            ]
            
            narrative = self.api_client.make_request(messages)
            return narrative if narrative else "Narrative generation unavailable"
            
        except Exception as e:
            logger.error(f"Error generating narrative: {str(e)}")
            return "Error generating narrative"

    def _generate_report(self, results: Dict[str, Any], narrative: str) -> None:
        """Generate comprehensive analysis report"""
        try:
            report_content = "# Data Analysis Report\n\n"
            
            # Add executive summary
            report_content += "## Executive Summary\n\n"
            report_content += narrative + "\n\n"
            
            # Add statistical findings
            report_content += "## Statistical Analysis\n\n"
            report_content += json.dumps(results['stats'], indent=2, default=str) + "\n\n"
            
            # Add insights
            report_content += "## Key Insights\n\n"
            report_content += results['insights'] + "\n\n"
            
            # Add visualizations
            report_content += "## Visualizations\n\n"
            for plot in self.plots:
                report_content += f"![{plot}]({plot})\n\n"
            
            # Save report
            report_path = self.config.output_dir / 'analysis_report.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"Analysis report generated at: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")

# Complete the main() function and add new components
def main():
    """Main execution function with enhanced error handling and workflow management"""
    try:
        # Parse command line arguments
        if len(sys.argv) != 2:
            print("Usage: python script.py <dataset_path>")
            sys.exit(1)

        file_path = sys.argv[1]
        
        # Initialize configuration
        config = AnalysisConfig(
            output_dir=Path("analysis_output"),
            color_blind_friendly=True,
            max_threads=4,
            visualization_dpi=300
        )
        
        # Initialize analyzers
        data_analyzer = DataAnalyzer(config)
        vision_analyzer = VisionAnalyzer(config)
        
        # Create workflow manager
        workflow_manager = AnalysisWorkflowManager(config, data_analyzer, vision_analyzer)
        
        # Execute analysis workflow
        workflow_manager.execute_workflow(file_path)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

class AnalysisWorkflowManager:
    """Manages the complete analysis workflow with multi-stage LLM integration"""
    
    def __init__(self, config: AnalysisConfig, data_analyzer: DataAnalyzer, 
                 vision_analyzer: VisionAnalyzer):
        self.config = config
        self.data_analyzer = data_analyzer
        self.vision_analyzer = vision_analyzer
        self.api_client = APIClient(config)

    def execute_workflow(self, file_path: str) -> None:
        """Execute the complete analysis workflow"""
        try:
            # Stage 1: Initial Analysis
            results = self.data_analyzer.analyze_dataset(file_path)
            
            # Stage 2: Vision Analysis
            vision_insights = self._analyze_visualizations()
            
            # Stage 3: Generate Enhanced Narrative
            narrative = self._generate_enhanced_narrative(results, vision_insights)
            
            # Stage 4: Create Final Report
            self._create_final_report(results, narrative, vision_insights)
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise

    def _analyze_visualizations(self) -> Dict[str, str]:
        """Analyze generated visualizations using vision capabilities"""
        vision_insights = {}
        for plot in self.data_analyzer.plots:
            plot_path = self.config.output_dir / plot
            if plot_path.exists():
                insight = self.vision_analyzer.analyze_visualization(plot_path)
                vision_insights[plot] = insight
        return vision_insights

    def _generate_enhanced_narrative(self, results: Dict[str, Any], 
                                   vision_insights: Dict[str, str]) -> str:
        """Generate enhanced narrative using multiple LLM calls"""
        try:
            # Initial narrative prompt
            initial_prompt = self._create_narrative_prompt(results, vision_insights)
            initial_narrative = self.api_client.make_request([
                {"role": "system", "content": "You are an expert data analyst and storyteller."},
                {"role": "user", "content": initial_prompt}
            ])

            # Refinement prompt
            refinement_prompt = f"""
            Review and enhance this narrative, focusing on:
            1. Clarity and flow
            2. Technical accuracy
            3. Business implications
            4. Actionable recommendations

            Original narrative:
            {initial_narrative}
            """
            
            refined_narrative = self.api_client.make_request([
                {"role": "system", "content": "You are an expert editor specializing in data analysis reports."},
                {"role": "user", "content": refinement_prompt}
            ])
            
            return refined_narrative if refined_narrative else initial_narrative
            
        except Exception as e:
            logger.error(f"Error generating enhanced narrative: {str(e)}")
            return "Error generating narrative"

    def _create_narrative_prompt(self, results: Dict[str, Any], 
                               vision_insights: Dict[str, str]) -> str:
        """Create dynamic narrative prompt based on analysis results"""
        key_stats = self._extract_key_statistics(results)
        viz_insights = "\n".join(vision_insights.values())
        
        return f"""
        Create a comprehensive data analysis narrative incorporating:

        Key Statistics:
        {json.dumps(key_stats, indent=2)}

        Visual Analysis Insights:
        {viz_insights}

        Requirements:
        1. Start with an executive summary highlighting key findings
        2. Discuss statistical significance and confidence levels
        3. Explain relationships between variables
        4. Highlight anomalies and their implications
        5. Provide specific, data-driven recommendations
        6. Use proper Markdown formatting with headers and emphasis
        7. Include references to specific visualizations
        8. End with clear next steps and implementation guidance

        Ensure the narrative is business-focused and actionable.
        """

    def _extract_key_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and summarize key statistics for narrative generation"""
        try:
            stats = results.get('stats', {})
            return {
                'summary': stats.get('basic_stats', {}).get('summary', {}),
                'correlations': self._summarize_correlations(stats),
                'outliers': self._summarize_outliers(stats),
                'normality': self._summarize_normality(stats)
            }
        except Exception as e:
            logger.error(f"Error extracting key statistics: {str(e)}")
            return {}

    def _create_final_report(self, results: Dict[str, Any], narrative: str, 
                           vision_insights: Dict[str, str]) -> None:
        """Create final report with enhanced formatting and organization"""
        try:
            report_path = self.config.output_dir / 'README.md'
            
            report_content = [
                "# Data Analysis Report\n",
                "## Executive Summary\n",
                narrative + "\n",
                "## Detailed Analysis\n",
                self._format_detailed_analysis(results) + "\n",
                "## Visual Insights\n",
                self._format_visual_insights(vision_insights) + "\n",
                "## Recommendations and Next Steps\n",
                self._generate_recommendations(results, vision_insights) + "\n",
                "## Technical Appendix\n",
                self._format_technical_details(results)
            ]
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_content))
                
            logger.info(f"Final report generated at: {report_path}")
            
        except Exception as e:
            logger.error(f"Error creating final report: {str(e)}")

    def _format_detailed_analysis(self, results: Dict[str, Any]) -> str:
        """Format detailed analysis section with proper Markdown"""
        try:
            return self.api_client.make_request([
                {"role": "system", "content": "You are a technical writer specializing in data analysis."},
                {"role": "user", "content": f"Create a detailed analysis section using these results: {json.dumps(results, default=str)}"}
            ]) or "Detailed analysis unavailable"
        except Exception as e:
            logger.error(f"Error formatting detailed analysis: {str(e)}")
            return "Error formatting analysis"

    def _format_visual_insights(self, vision_insights: Dict[str, str]) -> str:
        """Format visual insights with proper integration of images and analysis"""
        formatted_insights = []
        for plot_name, insight in vision_insights.items():
            formatted_insights.append(f"### {plot_name.replace('_', ' ').title()}\n")
            formatted_insights.append(f"![{plot_name}]({plot_name})\n")
            formatted_insights.append(f"{insight}\n")
        return "\n".join(formatted_insights)

    def _generate_recommendations(self, results: Dict[str, Any], 
                                vision_insights: Dict[str, str]) -> str:
        """Generate actionable recommendations based on analysis"""
        try:
            prompt = f"""
            Based on the analysis results and visual insights, generate specific,
            actionable recommendations. Include:
            1. Immediate actions
            2. Medium-term strategies
            3. Long-term considerations
            4. Risk factors and mitigation strategies
            5. Success metrics and KPIs

            Analysis Results:
            {json.dumps(results, default=str)}

            Visual Insights:
            {json.dumps(vision_insights, default=str)}
            """

            return self.api_client.make_request([
                {"role": "system", "content": "You are a strategic business consultant with expertise in data-driven decision making."},
                {"role": "user", "content": prompt}
            ]) or "Recommendations unavailable"
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return "Error generating recommendations"

if __name__ == "__main__":
    main()
