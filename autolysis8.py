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
from sklearn.cluster import KMeans
import time
from functools import wraps
import base64
from io import BytesIO
import cv2
from PIL import Image
import io

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
    """Enhanced configuration with vision settings"""
    max_retries: int = 3
    confidence_threshold: float = 0.95
    visualization_dpi: int = 300
    token_limit: int = 4000
    output_dir: Path = Path("output")
    time_limit: int = 180
    enable_vision: bool = True
    enable_multi_llm: bool = True
    vision_analysis_types: List[str] = None

    def __post_init__(self):
        self.vision_analysis_types = [
            'visualization_analysis',
            'image_content_analysis',
            'chart_detection'
        ]

class VisionAnalyzer:
    """Handles image analysis and vision tasks"""
    def __init__(self, api_client):
        self.api_client = api_client
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_visualization(self, image_path: str) -> Dict[str, Any]:
        """Analyze visualization using vision capabilities"""
        try:
            image_data = self.encode_image(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this visualization and provide insights about:"
                                   "\n1. Type of visualization"
                                   "\n2. Key patterns or trends"
                                   "\n3. Statistical observations"
                                   "\n4. Potential improvements"
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{image_data}"
                        }
                    ]
                }
            ]
            
            response = self.api_client.make_request(messages, model="gpt-4o-mini")
            return {
                'analysis': response,
                'image_path': image_path,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            print(f"Vision analysis failed: {str(e)}")
            return {'error': str(e)}

    def detect_chart_type(self, image_path: str) -> str:
        """Detect type of chart in visualization"""
        try:
            img = cv2.imread(image_path)
            # Basic chart detection logic
            # This could be enhanced with ML-based chart detection
            return "chart_type_detected"
        except Exception as e:
            return f"Chart detection failed: {str(e)}"

class APIClient:
    """Enhanced API client with vision support"""
    def __init__(self):
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise EnvironmentError("AIPROXY_TOKEN is not set")
        
        self.proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def make_request(self, messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> Optional[str]:
        """Make API request with enhanced error handling"""
        try:
            data = {
                "model":"gpt-4o-mini",
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

class AdvancedStatisticalMethods:
    """Advanced statistical analysis methods"""
    
    @staticmethod
    def trend_analysis(data: pd.Series) -> Dict[str, Any]:
        """Analyze trends using moving averages"""
        try:
            rolling_mean = data.rolling(window=3).mean()
            rolling_std = data.rolling(window=3).std()
            return {
                'trend': rolling_mean.tolist(),
                'volatility': rolling_std.tolist(),
                'direction': 'increasing' if rolling_mean.iloc[-1] > rolling_mean.iloc[0] else 'decreasing'
            }
        except Exception:
            return {}

    @staticmethod
    def hypothesis_testing(data1: pd.Series, data2: pd.Series) -> Dict[str, Any]:
        """Perform statistical hypothesis testing"""
        try:
            t_stat, p_value = stats.ttest_ind(data1.dropna(), data2.dropna())
            return {
                'test_type': 't_test',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except Exception:
            return {}

    @staticmethod
    def clustering_analysis(data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis"""
        try:
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(data)
            return {
                'n_clusters': 3,
                'cluster_sizes': np.bincount(clusters).tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist()
            }
        except Exception:
            return {}

class EnhancedVisualizationStrategy(ABC):
    """Enhanced base class for visualizations"""
    @abstractmethod
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        pass

    def add_annotations(self, ax, data: pd.Series, title: str):
        """Add statistical annotations to plot"""
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='g', linestyle='--', label=f'Median: {median_val:.2f}')
        ax.set_title(f"{title}\n(σ: {data.std():.2f})")
        ax.legend()

class EnhancedCorrelationHeatmap(EnhancedVisualizationStrategy):
    """Enhanced correlation heatmap"""
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        numeric_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(12, 8))
        
        g = sns.clustermap(
            numeric_df.corr(),
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        g.fig.suptitle(f"Correlation Analysis - {title}", y=1.02)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

class EnhancedDistributionPlot(EnhancedVisualizationStrategy):
    """Enhanced distribution plot"""
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        if n_cols == 0:
            return
            
        fig = plt.figure(figsize=(15, 5 * ((n_cols + 1) // 2)))
        for i, col in enumerate(numeric_cols, 1):
            ax = plt.subplot(((n_cols + 1) // 2), 2, i)
            
            sns.histplot(df[col], kde=True, ax=ax)
            self.add_annotations(ax, df[col], f"Distribution of {col}")
            
            skew = df[col].skew()
            kurt = df[col].kurtosis()
            ax.text(0.95, 0.95, f'Skewness: {skew:.2f}\nKurtosis: {kurt:.2f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.suptitle(f"Distribution Analysis - {title}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

class MultiLLMAnalyzer:
    """Handles multiple LLM interactions"""
    def __init__(self):
        self.api_client = APIClient()
        self.models = ["gpt-4o-mini"]
        
    def get_consensus_analysis(self, prompt: str) -> str:
        """Get analysis from multiple models"""
        responses = []
        for model in self.models:
            messages = [
                {"role": "system", "content": "You are a data analysis expert."},
                {"role": "user", "content": prompt}
            ]
            response = self.api_client.make_request(messages, model)
            if response:
                responses.append(response)
        
        if not responses:
            return "No analysis generated"
            
        return "\n\nConsensus Analysis:\n" + "\n".join(
            f"Analysis {i+1}:\n{response}" 
            for i, response in enumerate(responses)
        )

class EnhancedDataAnalyzer:
    """Enhanced data analyzer with vision capabilities"""
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.api_client = APIClient()
        self.vision_analyzer = VisionAnalyzer(self.api_client)
        self.multi_llm = MultiLLMAnalyzer()
        self.advanced_stats = AdvancedStatisticalMethods()
        self.visualization_strategies = [
            EnhancedCorrelationHeatmap(),
            EnhancedDistributionPlot()
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
            stats = self._compute_statistics(df)
            
            print("Generating visualizations...")
            self._generate_visualizations(df)
            
            print("Analyzing visualizations...")
            visual_insights = self._analyze_visualizations()
            
            if time.time() - start_time < self.config.time_limit - 30:
                print("Generating insights...")
                insights = self._generate_insights(df, stats, visual_insights)
                
                if time.time() - start_time < self.config.time_limit - 15:
                    print("Generating narrative...")
                    narrative = self._generate_narrative(df, stats, insights, visual_insights)
                    if narrative:
                        self._generate_readme(narrative, visual_insights)
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
        if not path.exists():
            raise ValueError(f"Invalid file path: {file_path}")
            
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            
        if df.empty:
            raise ValueError("Dataset is empty")
            
        return df

    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive statistics"""
        stats = {
            'basic_stats': self.advanced_stats.trend_analysis(df.select_dtypes(include=[np.number]).mean()),
            'clustering': self.advanced_stats.clustering_analysis(df.select_dtypes(include=[np.number]).fillna(0))
        }
        
        # Add correlation analysis
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            for col1 in numeric_df.columns:
                for col2 in numeric_df.columns:
                    if col1 < col2:
                        stats[f'hypothesis_{col1}_{col2}'] = self.advanced_stats.hypothesis_testing(
                            numeric_df[col1], numeric_df[col2]
                        )
        
        return stats

    def _generate_visualizations(self, df: pd.DataFrame):
        """Generate enhanced visualizations"""
        for i, strategy in enumerate(self.visualization_strategies, 1):
            print(f"Generating visualization {i}/{len(self.visualization_strategies)}...")
            viz_path = self.config.output_dir / f'visualization_{i}.png'
            strategy.create_visualization(df, viz_path, f"Analysis {i}")
            self.plots.append(viz_path.name)

    def _analyze_visualizations(self) -> Dict[str, Any]:
        """Analyze visualizations using vision capabilities"""
        visual_insights = {}
        if self.config.enable_vision:
            for plot in self.plots:
                plot_path = self.config.output_dir / plot
                analysis = self.vision_analyzer.analyze_visualization(str(plot_path))
                chart_type = self.vision_analyzer.detect_chart_type(str(plot_path))
                visual_insights[plot] = {
                    'analysis': analysis,
                    'chart_type': chart_type
                }
        return visual_insights

    def _generate_insights(self, df: pd.DataFrame, stats: Dict[str, Any],
                         visual_insights: Dict[str, Any]) -> str:
        """Generate comprehensive insights"""
        prompt = f"""
        Analyze this dataset:
        
        1. Dataset Overview:
        - Rows: {df.shape[0]}
        - Columns: {df.shape[1]}
        
        2. Statistical Analysis:
        {json.dumps(stats, indent=2, default=str)[:500]}
        
        3. Visual Analysis:
        {json.dumps(visual_insights, indent=2)[:500]}
        
        Provide:
        1. Key patterns and trends
        2. Statistical significance
        3. Visual insights
        4. Recommendations
        """

        return self.multi_llm.get_consensus_analysis(prompt)

    def _generate_narrative(self, df: pd.DataFrame, stats: Dict[str, Any],
                          insights: str, visual_insights: Dict[str, Any]) -> str:
        """Generate enhanced narrative"""
        story_template = f"""
        Create a comprehensive data story:

        Chapter 1: Executive Summary
        - Dataset: {df.shape[0]} rows, {df.shape[1]} columns
        - Key highlights

        Chapter 2: Data Profile
        - Types and distributions
        - Quality metrics

        Chapter 3: Statistical Insights
        {json.dumps(stats, indent=2, default=str)[:500]}

        Chapter 4: Visual Analysis
        {json.dumps(visual_insights, indent=2)[:500]}

        Chapter 5: Key Findings
        {insights[:500]}

        Chapter 6: Recommendations
        - Data-driven suggestions
        - Action items

        Format with clear sections and engaging narrative flow.
        """

        return self.multi_llm.get_consensus_analysis(story_template)

    def _generate_readme(self, narrative: str, visual_insights: Dict[str, Any]):
        """Generate enhanced README"""
        readme_content = "# Data Analysis Report\n\n"
        readme_content += narrative + "\n\n"
        
        readme_content += "## Visualizations\n\n"
        for plot in self.plots:
            readme_content += f"### {plot.replace('_', ' ').replace('.png', '').title()}\n"
            if plot in visual_insights:
                readme_content += f"Analysis: {visual_insights[plot]['analysis']}\n\n"
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
        config = AnalysisConfig(
            output_dir=Path(Path(file_path).stem),
            enable_vision=True,
            enable_multi_llm=True
        )
        
        analyzer = EnhancedDataAnalyzer(config)
        analyzer.analyze_dataset(file_path)

    except Exception as e:
        print(f"Program failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
