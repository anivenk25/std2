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

@dataclass
class AnalysisConfig:
    """Configuration settings for analysis"""
    max_retries: int = 3
    confidence_threshold: float = 0.95
    visualization_dpi: int = 300
    token_limit: int = 4000
    output_dir: Path = Path("output")

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
    """Handles statistical analysis of data"""
    def compute_advanced_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        stats = {
            'basic_stats': df.describe(),
            'missing_data': {
                'missing_counts': df.isnull().sum(),
                'missing_percentages': (df.isnull().sum() / len(df)) * 100
            }
        }
        
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats.update({
                'advanced_stats': {
                    'skewness': numeric_df.skew(),
                    'kurtosis': numeric_df.kurtosis(),
                    'correlations': numeric_df.corr()
                }
            })
        
        return stats

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
        
    def analyze_dataset(self, file_path: str):
        """Main method to analyze the dataset"""
        try:
            self._create_output_directory()
            df = self._load_and_validate_dataset(file_path)
            print(f"Successfully loaded dataset with shape: {df.shape}")

            # Generate initial analysis
            stats = self.stats_analyzer.compute_advanced_stats(df)
            print("\nGenerating visualizations and analysis...")
            
            # Generate visualizations
            self._generate_visualizations(df)
            
            # Generate insights
            insights = self._generate_insights(df, stats)
            
            # Generate narrative
            narrative = self._generate_narrative(df, stats, insights)
            
            if narrative:
                self._generate_readme(narrative)
                
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            traceback.print_exc()

    def _create_output_directory(self):
        """Create or clean output directory"""
        if self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True)

    def _load_and_validate_dataset(self, file_path: str) -> pd.DataFrame:
        """Load and validate dataset with error handling"""
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
        """Generate visualizations using all strategies"""
        for i, strategy in enumerate(self.visualization_strategies):
            viz_path = self.config.output_dir / f'visualization_{i}.png'
            strategy.create_visualization(
                df, 
                viz_path, 
                f"Analysis {i+1}"
            )
            self.plots.append(viz_path.name)

    def _generate_insights(self, df: pd.DataFrame, stats: Dict[str, Any]) -> str:
        """Generate insights using LLM"""
        prompt = f"""
        Analyze this dataset based on the following information:

        1. Dataset Statistics:
        {stats['basic_stats'].to_string()}

        2. Missing Data Analysis:
        {stats['missing_data']['missing_counts'].to_string()}

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
        """Generate narrative story from analysis"""
        subject = self._determine_subject(df)
        
        story_prompt = f"""
        Create an engaging narrative about this {subject} dataset:

        Key Statistics:
        {stats['basic_stats'].to_string()}

        Insights:
        {insights}

        Requirements:
        1. Create a compelling story that explains the data journey
        2. Include specific numbers and findings
        3. Make it engaging and memorable
        4. Include implications and recommendations
        5. Use clear sections and structure

        The narrative should flow naturally while incorporating technical insights.
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
        """Determine dataset subject using column analysis"""
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
        """Generate README with narrative and visualizations"""
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
