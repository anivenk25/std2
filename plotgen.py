import os
import sys
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
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
    max_retries: int = 3
    confidence_threshold: float = 0.95
    visualization_dpi: int = 300
    output_dir: Path = Path("output")
    time_limit: int = 120

class APIClient:
    def __init__(self):
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise EnvironmentError("AIPROXY_TOKEN is not set")
        self.proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def make_request(self, messages: List[Dict[str, str]], detail: str = "low") -> Optional[str]:
        try:
            data = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "detail": detail
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

class PlotManager:
    @staticmethod
    def determine_plots(df: pd.DataFrame, api_client: APIClient) -> List[Dict[str, Any]]:
        prompt = f"""
        Given this dataset:
        - Columns: {df.columns.tolist()}
        - Data types: {df.dtypes.to_dict()}
        - Sample: {df.head(2).to_dict()}

        Suggest the most insightful visualizations as a JSON list with:
        - plot_type: string (e.g., 'scatter', 'heatmap', 'bar', 'box', 'violin', 'line')
        - columns: list of column names
        - title: string
        - description: string explaining insights to look for

        Focus on key relationships and patterns.
        """
        
        response = api_client.make_request([
            {"role": "system", "content": "You are a data visualization expert."},
            {"role": "user", "content": prompt}
        ], detail="low")
        
        try:
            return json.loads(response)
        except:
            return [{"plot_type": "heatmap", "columns": df.select_dtypes(include=[np.number]).columns.tolist(), "title": "Correlation Heatmap"}]

    @staticmethod
    def create_plot(plot_spec: Dict[str, Any], df: pd.DataFrame, output_dir: Path) -> str:
        plt.figure(figsize=(10, 6))
        
        if plot_spec["plot_type"] == "heatmap":
            sns.heatmap(df[plot_spec["columns"]].corr(), annot=True, cmap='coolwarm')
        elif plot_spec["plot_type"] == "scatter":
            sns.scatterplot(data=df, x=plot_spec["columns"][0], y=plot_spec["columns"][1])
        elif plot_spec["plot_type"] == "bar":
            df[plot_spec["columns"]].plot(kind='bar')
        elif plot_spec["plot_type"] == "box":
            df[plot_spec["columns"]].boxplot()
        elif plot_spec["plot_type"] == "violin":
            sns.violinplot(data=df[plot_spec["columns"]])
        elif plot_spec["plot_type"] == "line":
            df[plot_spec["columns"]].plot(kind='line')
            
        plt.title(plot_spec["title"])
        plt.tight_layout()
        
        plot_path = output_dir / f"{plot_spec['title'].lower().replace(' ', '_')}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return str(plot_path)

class DataAnalyzer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.api_client = APIClient()
        self.plots = []
        
    def analyze_plots(self, plot_paths: List[str], df: pd.DataFrame) -> str:
        plot_analysis_prompt = f"""
        Analyze these visualizations:
        {[str(p) for p in plot_paths]}
        
        Dataset summary:
        {df.describe().to_string()}
        
        Provide insights about:
        1. Patterns and trends
        2. Correlations and relationships
        3. Anomalies or interesting findings
        4. Key metrics and their implications
        """
        
        return self.api_client.make_request([
            {"role": "system", "content": "You are a data visualization analyst."},
            {"role": "user", "content": plot_analysis_prompt}
        ], detail="low")

    @timeit
    def analyze_dataset(self, file_path: str):
        try:
            df = pd.read_csv(file_path)
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get plot suggestions from LLM
            plot_specs = PlotManager.determine_plots(df, self.api_client)
            
            # Generate plots
            plot_paths = []
            for spec in plot_specs:
                plot_path = PlotManager.create_plot(spec, df, self.config.output_dir)
                plot_paths.append(plot_path)
            
            # Analyze plots
            plot_insights = self.analyze_plots(plot_paths, df)
            
            # Generate narrative
            narrative = self.generate_narrative(df, plot_insights)
            
            # Save results
            self.save_results(narrative, plot_paths)
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            traceback.print_exc()

    def generate_narrative(self, df: pd.DataFrame, insights: str) -> str:
        narrative_prompt = f"""
        Create an engaging story about this data analysis:
        
        Dataset Info:
        {df.describe().to_string()}
        
        Analysis Insights:
        {insights}
        
        Create a compelling narrative that:
        1. Explains key findings and patterns
        2. Uses specific numbers and metrics
        3. Provides actionable recommendations
        """
        
        return self.api_client.make_request([
            {"role": "system", "content": "You are a data storyteller."},
            {"role": "user", "content": narrative_prompt}
        ], detail="low") or ""

    def save_results(self, narrative: str, plot_paths: List[str]):
        readme_content = "# Data Analysis Results\n\n"
        readme_content += narrative + "\n\n"
        readme_content += "## Visualizations\n\n"
        
        for plot_path in plot_paths:
            readme_content += f"![{Path(plot_path).stem}]({plot_path})\n\n"
            
        with open(self.config.output_dir / 'README.md', 'w') as f:
            f.write(readme_content)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    config = AnalysisConfig(output_dir=Path(Path(file_path).stem))
    analyzer = DataAnalyzer(config)
    analyzer.analyze_dataset(file_path)

if __name__ == "__main__":
    main()
