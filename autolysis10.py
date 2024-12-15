import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import traceback
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio
import aiohttp
import time
from functools import wraps
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    """Decorator to measure function execution time."""
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
    """Configuration settings for analysis."""
    max_retries: int = 3
    confidence_threshold: float = 0.95
    visualization_dpi: int = 300
    token_limit: int = 4000
    output_dir: Path = Path("output")
    time_limit: int = 180  # 3 minutes in seconds

class AsyncAPIClient:
    """Handles asynchronous API communication with LLM service."""
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise EnvironmentError("AIPROXY_TOKEN is not set")

    async def fetch(self, session, endpoint, payload):
        """Fetch data from the API asynchronously."""
        url = f"{self.base_url}/{endpoint}"
        async with session.post(url, json=payload) as response:
            return await response.json()

    async def call_api(self, endpoint, payload):
        """Public method to call API asynchronously."""
        async with aiohttp.ClientSession() as session:
            return await self.fetch(session, endpoint, payload)

class StatisticalMethods:
    """Collection of statistical analysis methods."""
    
    @staticmethod
    def basic_stats(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics."""
        return {
            'summary': data.describe(),
            'missing': data.isnull().sum(),
            'dtypes': data.dtypes
        }

    @staticmethod
    def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if data.isnull().sum().sum() > 0:
            print("Handling missing values...")
            data = data.dropna()  # or use fillna() with a strategy
        return data

    @staticmethod
    def normality_test(data: pd.Series) -> Dict[str, Any]:
        """Perform normality test."""
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
        """Detect outliers using IQR method."""
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
        """Perform PCA for dimensionality reduction."""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(scaled_data)
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': transformed.tolist()
        }

class VisualizationStrategy:
    """Abstract base class for visualization strategies."""
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        """Create a visualization."""
        raise NotImplementedError

class CorrelationHeatmap(VisualizationStrategy):
    """Generate a correlation heatmap."""
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        numeric_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(12, 8))
        plt.title(f"Correlation Heatmap - {title}")
        plt.imshow(numeric_df.corr(), cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(numeric_df.columns)), numeric_df.columns, rotation=90)
        plt.yticks(range(len(numeric_df.columns)), numeric_df.columns)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

class DistributionPlot(VisualizationStrategy):
    """Generate distribution plots."""
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        if n_cols == 0:
            return
            
        plt.figure(figsize=(15, 5 * ((n_cols + 1) // 2)))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(((n_cols + 1) // 2), 2, i)
            plt.hist(df[col], bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
        
        plt.suptitle(f"Distribution Analysis - {title}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

class DataAnalyzer:
    """Enhanced data analyzer with comprehensive analysis capabilities."""
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.api_client = AsyncAPIClient(base_url="https://aiproxy.sanand.workers.dev/openai/v1/chat/completions")
        self.stats_methods = StatisticalMethods()
        self.visualization_strategies = [
            CorrelationHeatmap(),
            DistributionPlot()
        ]
        self.plots: List[str] = []
        
    @timeit
    async def analyze_dataset(self, file_path: str):
        """Main method to analyze the dataset."""
        try:
            start_time = time.time()
            self._create_output_directory()
            df = await self._load_and_validate_dataset(file_path)
            print(f"Successfully loaded dataset with shape: {df.shape}")

            # Handle missing values
            df = self.stats_methods.handle_missing_values(df)

            stats = self.stats_methods.basic_stats(df)
            print("\nGenerating visualizations and analysis...")
            
            await self._generate_visualizations(df)
            
            insights = await self._generate_insights(df, stats)
            
            if time.time() - start_time < self.config.time_limit:
                narrative = await self._generate_narrative(df, stats, insights)
                if narrative:
                    self._generate_readme(narrative)
            else:
                print("Time limit reached, skipping narrative generation")
                
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            traceback.print_exc()

    def _create_output_directory(self):
        """Create output directory for results."""
        if self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True)

    async def _load_and_validate_dataset(self, file_path: str) -> pd.DataFrame:
        """Load and validate the dataset."""
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

    async def _generate_visualizations(self, df: pd.DataFrame):
        """Generate visualizations for the dataset."""
        for i, strategy in enumerate(self.visualization_strategies):
            viz_path = self.config.output_dir / f'visualization_{i}.png'
            strategy.create_visualization(df, viz_path, f"Analysis {i+1}")
            self.plots.append(viz_path.name)

    async def _generate_insights(self, df: pd.DataFrame, stats: Dict[str, Any]) -> str:
        """Generate insights based on statistical analysis."""
        prompt = f"""
        Analyze this dataset based on the following information:

        1. Dataset Statistics:
        {stats['summary'].to_string()}

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

        insights = await self.api_client.call_api("analyze", messages)
        if insights:
            print("\nKey Insights Generated")
            return insights
        return ""

    async def _generate_narrative(self, df: pd.DataFrame, stats: Dict[str, Any], insights: str) -> str:
        """Generate a narrative based on the analysis."""
        subject = self._determine_subject(df)
        
        story_prompt = f"""
        Create an engaging narrative about this {subject} dataset:

        Key Statistics:
        {stats['summary'].to_string()}

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

        narrative = await self.api_client.call_api("narrate", messages)
        if narrative:
            print("\nNarrative Generated")
            return narrative
        return ""

    def _determine_subject(self, df: pd.DataFrame) -> str:
        """Determine the subject of the dataset based on column names."""
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
        """Generate a README file with the narrative and visualizations."""
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
    """Main execution function."""
    try:
        if len(sys.argv) != 2:
            print("Usage: python script.py dataset.csv")
            sys.exit(1)

        file_path = sys.argv[1]
        config = AnalysisConfig(output_dir=Path(Path(file_path).stem))
        analyzer = DataAnalyzer(config)
        asyncio.run(analyzer.analyze_dataset(file_path))

    except Exception as e:
        print(f"Program failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
