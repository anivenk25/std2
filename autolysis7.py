import os
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import traceback
from typing import Optional, Dict, Any, List
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
    """Decorator to measure function execution time"""
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
    time_limit: int = 180  # 3 minutes

class APIClient:
    """Handles API communication"""
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
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
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
    """Statistical analysis methods"""
    
    @staticmethod
    def basic_stats(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics"""
        return {
            'summary': data.describe(),
            'missing': data.isnull().sum(),
            'dtypes': data.dtypes
        }

    @staticmethod
    def normality_test(data: pd.Series) -> Dict[str, Any]:
        """Perform normality test"""
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
        """Detect outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'outlier_values': outliers.tolist()
        }

class VisualizationStrategy(ABC):
    """Abstract base class for visualization strategies"""
    @abstractmethod
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        pass

class CorrelationHeatmap(VisualizationStrategy):
    """Creates correlation heatmap"""
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        numeric_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(f"Correlation Heatmap - {title}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

class DistributionPlot(VisualizationStrategy):
    """Creates distribution plots"""
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
    """Handles statistical analysis"""
    def __init__(self):
        self.methods = StatisticalMethods()
        self.api_client = APIClient()
        
    @timeit
    def select_analysis_methods(self, df: pd.DataFrame) -> List[str]:
        """Select appropriate statistical methods"""
        return ['basic_stats', 'normality_test', 'outlier_detection']

    @timeit
    def compute_advanced_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistical analysis"""
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
                
            except Exception as e:
                print(f"Error in {method}: {str(e)}")
                continue
                
        return results

class DataAnalyzer:
    """Main data analyzer class"""
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
        """Main analysis method"""
        try:
            start_time = time.time()
            self._create_output_directory()
            
            print("Loading dataset...")
            df = self._load_and_validate_dataset(file_path)
            print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

            print("Computing statistical analysis...")
            stats = self.stats_analyzer.compute_advanced_stats(df)
            
            print("Generating visualizations...")
            self._generate_visualizations(df)
            
            if time.time() - start_time < self.config.time_limit - 30:
                print("Generating insights...")
                insights = self._generate_insights(df, stats)
                
                if time.time() - start_time < self.config.time_limit - 15:
                    print("Generating narrative...")
                    narrative = self._generate_narrative(df, stats, insights)
                    if narrative:
                        self._generate_readme(narrative)
                else:
                    print("Skipping narrative generation due to time constraint")
            else:
                print("Skipping insights and narrative generation due to time constraint")
                
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            traceback.print_exc()

    def _create_output_directory(self):
        """Create output directory"""
        if self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True)

    def _load_and_validate_dataset(self, file_path: str) -> pd.DataFrame:
        """Load and validate dataset"""
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
        """Generate visualizations"""
        for i, strategy in enumerate(self.visualization_strategies, 1):
            print(f"Generating visualization {i}/{len(self.visualization_strategies)}...")
            viz_path = self.config.output_dir / f'visualization_{i}.png'
            strategy.create_visualization(df, viz_path, f"Analysis {i}")
            self.plots.append(viz_path.name)

    def _generate_insights(self, df: pd.DataFrame, stats: Dict[str, Any]) -> str:
        """Generate insights"""
        prompt = f"""
        Analyze this dataset:
        Rows: {df.shape[0]}
        Columns: {df.shape[1]}
        
        Key findings:
        {json.dumps(stats.get('basic_stats', {}), indent=2, default=str)[:500]}
        
        Provide key insights in bullet points.
        """

        messages = [
            {"role": "system", "content": "You are a data analyst. Be concise."},
            {"role": "user", "content": prompt}
        ]

        return self.api_client.make_request(messages) or "No insights generated"

    def _generate_narrative(self, df: pd.DataFrame, stats: Dict[str, Any], 
                          insights: str) -> str:
        """Generate narrative"""
        story_prompt = f"""
        Create an engaging narrative/ story about this dataset make it like a movie / great novel :

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

        return self.api_client.make_request(messages) or "No narrative generated"

    def _generate_readme(self, narrative: str):
        """Generate README file"""
        readme_content = "# Data Analysis Report\n\n"
        readme_content += narrative + "\n\n"
        
        readme_content += "## Visualizations\n\n"
        for plot in self.plots:
            readme_content += f"![{plot}]({plot})\n\n"
            
        readme_path = self.config.output_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"\nAnalysis report generated at: {readme_path}")

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
