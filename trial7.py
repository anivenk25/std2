import os
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple
import traceback
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import networkx as nx
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class InnovativeDataAnalyzer:
    def __init__(self):
        """Initialize the analyzer with advanced capabilities."""
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise EnvironmentError("AIPROXY_TOKEN environment variable not set")
        
        self.proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.insights = []
        self.visualizations = []
        self.story_sections = []
        self.analysis_context = {}
        
    def _make_llm_request(self, messages: List[Dict[str, str]], 
                         temperature: float = 0.7) -> Optional[str]:
        """Make an efficient request to the LLM API."""
        try:
            data = {
                "model": "gpt-4",
                "messages": messages,
                "temperature": temperature
            }
            response = requests.post(self.proxy_url, headers=self.headers, 
                                   json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"LLM request failed: {str(e)}")
            return None

    def _detect_data_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect sophisticated patterns in the dataset."""
        patterns = {
            'temporal': False,
            'categorical': [],
            'numerical': [],
            'text_columns': [],
            'correlations': {},
            'anomalies': {},
            'clusters': {}
        }
        
        # Detect temporal patterns
        date_columns = df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            patterns['temporal'] = True
            patterns['date_columns'] = date_columns.tolist()
        
        # Analyze numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) >= 2:
            # Correlation analysis
            corr_matrix = df[num_cols].corr()
            strong_corr = np.where(np.abs(corr_matrix) > 0.7)
            patterns['correlations'] = {
                f"{num_cols[i]}_{num_cols[j]}": corr_matrix.iloc[i,j]
                for i, j in zip(*strong_corr) if i < j
            }
            
            # Anomaly detection using DBSCAN
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df[num_cols])
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(data_scaled)
            patterns['anomalies'] = {
                'count': sum(clusters == -1),
                'percentage': sum(clusters == -1) / len(clusters) * 100
            }
            
            # Clustering analysis
            kmeans = KMeans(n_clusters=min(5, len(df)))
            cluster_labels = kmeans.fit_predict(data_scaled)
            patterns['clusters'] = {
                'n_clusters': len(np.unique(cluster_labels)),
                'sizes': pd.Series(cluster_labels).value_counts().to_dict()
            }
        
        # Analyze categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        patterns['categorical'] = [
            {
                'column': col,
                'unique_values': df[col].nunique(),
                'top_categories': df[col].value_counts().head(5).to_dict()
            }
            for col in cat_cols
        ]
        
        # Text analysis for string columns
        for col in cat_cols:
            if df[col].str.len().mean() > 50:  # Long text fields
                patterns['text_columns'].append(col)
                
        return patterns

    def _create_advanced_visualization(self, df: pd.DataFrame, 
                                    viz_type: str, **kwargs) -> Optional[str]:
        """Create sophisticated visualizations based on data patterns."""
        try:
            plt.figure(figsize=(12, 8))
            
            if viz_type == "correlation_network":
                return self._create_correlation_network(df, **kwargs)
            elif viz_type == "anomaly_scatter":
                return self._create_anomaly_visualization(df, **kwargs)
            elif viz_type == "cluster_analysis":
                return self._create_cluster_visualization(df, **kwargs)
            elif viz_type == "temporal_analysis":
                return self._create_temporal_visualization(df, **kwargs)
            elif viz_type == "text_analysis":
                return self._create_text_visualization(df, **kwargs)
            
            plt.close()
            return None
        except Exception as e:
            print(f"Visualization creation failed: {str(e)}")
            plt.close()
            return None
    
    def _create_correlation_network(self, df: pd.DataFrame, **kwargs) -> Optional[str]:
        """Create a network visualization of correlations."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        G = nx.Graph()
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.3:
                    G.add_edge(corr.columns[i], corr.columns[j], weight=abs(corr.iloc[i, j]))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=8)
        filename = 'network_viz.png'
        plt.savefig(filename)
        plt.close()
        return filename

    def _create_anomaly_visualization(self, df: pd.DataFrame, **kwargs) -> Optional[str]:
        """Create visualization of anomalies using DBSCAN."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[numeric_cols])
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(data_scaled)
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_scaled)
        fig = px.scatter(x=data_2d[:, 0], y=data_2d[:, 1], color=clusters, title="Anomaly Detection Visualization")
        filename = 'anomalies.html'
        fig.write_html(filename)
        return filename

    def _create_cluster_visualization(self, df: pd.DataFrame, **kwargs) -> Optional[str]:
        """Create a cluster analysis visualization."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[numeric_cols])
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(data_scaled)
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_scaled)
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis')
        plt.title("Cluster Analysis")
        filename = 'cluster_analysis.png'
        plt.savefig(filename)
        plt.close()
        return filename

    def _create_temporal_visualization(self, df: pd.DataFrame, **kwargs) -> Optional[str]:
        """Create a temporal analysis visualization."""
        date_cols = kwargs.get('date_columns', [])
        if not date_cols:
            return None
        for date_col in date_cols:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col).resample('M').size().plot()
            plt.title(f"Temporal Analysis of {date_col}")
            filename = f'temporal_{date_col}.png'
            plt.savefig(filename)
            plt.close()
            return filename

    def _create_text_visualization(self, df: pd.DataFrame, **kwargs) -> Optional[str]:
        """Create a word cloud from text data."""
        text_col = kwargs.get('text_column', None)
        if not text_col or not pd.api.types.is_string_dtype(df[text_col]):
            return None
        text = ' '.join(df[text_col].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        filename = f'wordcloud_{text_col}.png'
        plt.savefig(filename)
        plt.close()
        return filename

    def _generate_story(self, df: pd.DataFrame, patterns: Dict[str, Any]) -> str:
        """Generate a compelling narrative about the data."""
        prompt = f"""
        Create a narrative that explains the following data patterns:
        - Temporal patterns: {patterns.get('temporal', False)}
        - Strong correlations: {patterns.get('correlations', {})}
        - Anomalies detected: {patterns.get('anomalies', {})}
        - Clusters found: {patterns.get('clusters', {})}
        - Text insights: {patterns.get('text_columns', [])}
        
        Use metaphors and storytelling techniques to make the insights engaging.
        """
        messages = [
            {"role": "system", "content": "You are a creative data storyteller."},
            {"role": "user", "content": prompt}
        ]
        story = self._make_llm_request(messages)
        return story if story else "No story generated."

    def analyze_dataset(self, file_path: str):
        """Main method to analyze the dataset."""
        try:
            df = pd.read_csv(file_path)
            patterns = self._detect_data_patterns(df)
            self.analysis_context = patterns
            
            # Create visualizations
            if patterns['correlations']:
                self.visualizations.append(self._create_advanced_visualization(df, "correlation_network"))
            if patterns['anomalies']:
                self.visualizations.append(self._create_advanced_visualization(df, "anomaly_scatter"))
            if patterns['clusters']:
                self.visualizations.append(self._create_advanced_visualization(df, "cluster_analysis"))
            if patterns['temporal']:
                self.visualizations.append(self._create_advanced_visualization(df, "temporal_analysis", date_columns=patterns['date_columns']))
            for text_col in patterns['text_columns']:
                self.visualizations.append(self._create_advanced_visualization(df, "text_analysis", text_column=text_col))
            
            # Generate story
            story = self._generate_story(df, patterns)
            
            # Save results
            self._save_results(story)
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            traceback.print_exc()

    def _save_results(self, story: str):
        """Save analysis results to README.md."""
        markdown = f"""# Data Story: A Journey Through the Numbers

{story}

## Visualizations

"""
        for viz in self.visualizations:
            if viz.endswith('.html'):
                markdown += f"- [Interactive Visualization]({viz})\n"
            else:
                markdown += f"![Visualization]({viz})\n"

        with open("README.md", "w") as f:
            f.write(markdown)

def main():
    analyzer = InnovativeDataAnalyzer()
    csv_files = list(Path.cwd().glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError("No CSV files found")
    analyzer.analyze_dataset(str(csv_files[0]))

if __name__ == "__main__":
    main()
