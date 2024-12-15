import os
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import traceback
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys
import shutil
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@dataclass
class AnalysisConfig:
    """Configuration settings for analysis"""
    max_retries: int = 3
    confidence_threshold: float = 0.95
    visualization_dpi: int = 300
    token_limit: int = 4000
    output_dir: Path = Path("output")
    time_limit: int = 180  # 3 minutes in seconds

class APIClient:
    """Handles API communication with LLM service"""
    def __init__(self):
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise EnvironmentError("AIPROXY_TOKEN is not set")
        
        self.proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def make_request(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make API request with error handling"""
        try:
            data = {
                "model": "gpt-4o-mini",
                "messages": messages
            }
            response = requests.post(
                self.proxy_url, 
                headers=self.headers, 
                json=data, 
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"API request failed: {str(e)}")
            return None

class StatisticalMethods:
    """Collection of statistical analysis methods"""
    
    @staticmethod
    def basic_stats(data: pd.DataFrame) -> Dict[str, Any]:
        return {
            'summary': data.describe(),
            'missing': data.isnull().sum(),
            'dtypes': data.dtypes
        }

    @staticmethod
    def normality_test(data: pd.Series) -> Dict[str, Any]:
        if len(data) < 3:
            return {'error': 'Insufficient data'}
        statistic, p_value = stats.normaltest(data.dropna())
        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }

    @staticmethod
    def outlier_detection(data: pd.Series) -> Dict[str, Any]:
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'outlier_values': outliers.tolist()
        }

    @staticmethod
    def dimension_reduction(data: pd.DataFrame, n_components: int = 2) -> Dict[str, Any]:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(scaled_data)
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': transformed.tolist()
        }

class VisualizationStrategy(ABC):
    """Abstract base class for visualization strategies"""
    @abstractmethod
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        pass

class CorrelationHeatmap(VisualizationStrategy):
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        numeric_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(f"Correlation Heatmap - {title}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

class DistributionPlot(VisualizationStrategy):
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        if n_cols == 0:
            return
            
        plt.figure(figsize=(15, 5 * ((n_cols + 1) // 2)))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(((n_cols + 1) // 2), 2, i)
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
        
        plt.suptitle(f"Distribution Analysis - {title}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

class StatisticalAnalyzer:
    """Enhanced statistical analyzer with method selection"""
    def __init__(self):
        self.methods = StatisticalMethods()
        self.api_client = APIClient()
        
    @timeit
    def select_analysis_methods(self, df: pd.DataFrame) -> List[str]:
        """Use LLM to select appropriate statistical methods"""
        # Convert dtypes to strings for JSON serialization
        data_description = {
            'shape': list(df.shape),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
        }
        
        prompt = f"""
        Given the following dataset characteristics:
        Shape: {data_description['shape']}
        Data Types: {json.dumps(data_description['dtypes'], indent=2)}
        Missing Values: {json.dumps(data_description['missing_values'], indent=2)}
        Numeric Columns: {json.dumps(data_description['numeric_columns'], indent=2)}
        
        Available statistical methods:
        1. basic_stats: Basic statistical summary
        2. normality_test: Test for normal distribution
        3. outlier_detection: Identify outliers using IQR
        4. dimension_reduction: PCA for dimensionality reduction
        
        Select the most appropriate methods considering:
        - Dataset size and characteristics
        - Time constraint (analysis should complete within 3 minutes)
        - Data types present
        
        Return a list of method names to apply.
        """
        
        messages = [
            {"role": "system", "content": "You are a statistical analysis expert."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.api_client.make_request(messages)
        if not response:
            return ['basic_stats']
            
        methods = []
        if 'basic_stats' in response.lower():
            methods.append('basic_stats')
        if 'normality' in response.lower():
            methods.append('normality_test')
        if 'outlier' in response.lower():
            methods.append('outlier_detection')
        if 'dimension' in response.lower() or 'pca' in response.lower():
            methods.append('dimension_reduction')
            
        return methods

    @timeit
    def compute_advanced_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistical analysis based on selected methods"""
        selected_methods = self.select_analysis_methods(df)
        results = {}
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        for method in selected_methods:
            try:
                if method == 'basic_stats':
                    results['basic_stats'] = self.methods.basic_stats(df)
                
                elif method == 'normality_test' and not numeric_df.empty:
                    results['normality_tests'] = {
                        col: self.methods.normality_test(numeric_df[col])
                        for col in numeric_df.columns
                    }
                
                elif method == 'outlier_detection' and not numeric_df.empty:
                    results['outlier_analysis'] = {
                        col: self.methods.outlier_detection(numeric_df[col])
                        for col in numeric_df.columns
                    }
                
                elif method == 'dimension_reduction' and not numeric_df.empty:
                    if numeric_df.shape[1] > 2:
                        results['dimension_reduction'] = self.methods.dimension_reduction(numeric_df)
                
            except Exception as e:
                print(f"Error in {method}: {str(e)}")
                continue
                
        return results

class DataAnalyzer:
    """Enhanced data analyzer with comprehensive analysis capabilities"""
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.api_client = APIClient()
        self.stats_analyzer = StatisticalAnalyzer()
        self.visualization_strategies = [
            CorrelationHeatmap(),
            DistributionPlot()
        ]
        self.plots: List[str] = []
        
    @timeit
    def analyze_dataset(self, file_path: str):
        """Main method to analyze the dataset"""
        try:
            start_time = time.time()
            self._create_output_directory()
            df = self._load_and_validate_dataset(file_path)
            print(f"Successfully loaded dataset with shape: {df.shape}")

            stats = self.stats_analyzer.compute_advanced_stats(df)
            print("\nGenerating visualizations and analysis...")
            
            self._generate_visualizations(df)
            
            insights = self._generate_insights(df, stats)
            
            if time.time() - start_time < self.config.time_limit:
                narrative = self._generate_narrative(df, stats, insights)
                if narrative:
                    self._generate_readme(narrative)
            else:
                print("Time limit reached, skipping narrative generation")
                
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            traceback.print_exc()

    def _create_output_directory(self):
        if self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True)

    def _load_and_validate_dataset(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists() or not path.is_file() or path.suffix.lower() != '.csv':
            raise ValueError(f"Invalid file path: {file_path}")
            
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            
        if df.empty:
            raise ValueError("Dataset is empty")
            
        return df

    def _generate_visualizations(self, df: pd.DataFrame):
        for i, strategy in enumerate(self.visualization_strategies):
            viz_path = self.config.output_dir / f'visualization_{i}.png'
            strategy.create_visualization(df, viz_path, f"Analysis {i+1}")
            self.plots.append(viz_path.name)

    def _generate_insights(self, df: pd.DataFrame, stats: Dict[str, Any]) -> str:
        prompt = f"""
        Analyze this dataset based on the following information:

        1. Dataset Statistics:
        {stats['basic_stats']['summary'].to_string()}

        2. Advanced Analysis:
        {json.dumps(stats, indent=2, default=str)}

        Please provide:
        1. Key patterns and trends
        2. Statistical findings
        3. Notable relationships between variables
        4. Distribution insights
        5. Recommendations

        Format with clear headers and bullet points.
        """

        messages = [
            {"role": "system", "content": "You are a data analyst specializing in statistical analysis."},
            {"role": "user", "content": prompt}
        ]

        insights = self.api_client.make_request(messages)
        if insights:
            print("\nKey Insights Generated")
            return insights
        return ""

    def _generate_narrative(self, df: pd.DataFrame, stats: Dict[str, Any], insights: str) -> str:
        subject = self._determine_subject(df)
        
        story_prompt = f"""
        Create an engaging narrative about this {subject} dataset:

        Key Statistics:
        {stats['basic_stats']['summary'].to_string()}

        Insights:
        {insights}

        Requirements:
        1. Create a compelling story that explains the data journey
        2. Include specific numbers and findings
        3. Make it engaging and memorable
        4. Include implications and recommendations
        5. Use clear sections and structure
        """

        messages = [
            {"role": "system", "content": "You are a data storyteller who transforms analysis into engaging narratives."},
            {"role": "user", "content": story_prompt}
        ]

        narrative = self.api_client.make_request(messages)
        if narrative:
            print("\nNarrative Generated")
            return narrative
        return ""

    def _determine_subject(self, df: pd.DataFrame) -> str:
        columns_str = ' '.join(df.columns.str.lower())
        common_subjects = {
            'book': ['book', 'author', 'title', 'publisher'],
            'movie': ['movie', 'film', 'director', 'actor'],
            'sales': ['sale', 'revenue', 'product', 'customer'],
            'financial': ['price', 'cost', 'revenue', 'profit']
        }
        
        for subject, keywords in common_subjects.items():
            if any(keyword in columns_str for keyword in keywords):
                return subject
                
        return "dataset"

    def _generate_readme(self, narrative: str):
        readme_content = "# Data Analysis Narrative\n\n"
        readme_content += narrative + "\n\n"
        
        readme_content += "## Visualizations\n\n"
        for plot in self.plots:
            readme_content += f"![{plot}]({plot})\n\n"
            
        readme_path = self.config.output_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"\nREADME.md generated at: {readme_path}")

def main():
    """Main execution function"""
    try:
        if len(sys.argv) != 2:
            print("Usage: python script.py dataset.csv")
            sys.exit(1)

        file_path = sys.argv[1]
        config = AnalysisConfig(output_dir=Path(Path(file_path).stem))
        analyzer = DataAnalyzer(config)
        analyzer.analyze_dataset(file_path)

    except Exception as e:
        print(f"Program failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
