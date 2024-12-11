#!/usr/bin/env python3
"""
Automated Dataset Analysis Script with LLM Integration
Usage: uv run autolysis.py dataset.csv
"""

import os
import sys
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class DatasetAnalyzer:
    def __init__(self, token: str):
        """Initialize the analyzer with API token."""
        self.token = token
        self.proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.insights = []
        self.visualizations = []
        self.story_sections = []

    def _ask_llm(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make a request to the LLM API with error handling."""
        try:
            data = {
                "model": "gpt-4",
                "messages": messages,
                "temperature": 0.7
            }
            response = requests.post(self.proxy_url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"LLM request failed: {str(e)}")
            return None

    def _validate_csv(self, file_path: str) -> pd.DataFrame:
        """Validate and load CSV file with comprehensive error handling."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError("Unable to read CSV file with supported encodings")

        if df.empty:
            raise ValueError("CSV file is empty")

        return df

    def _get_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
        numeric_df = df.select_dtypes(include=[np.number])

        profile = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict() if not numeric_df.empty else {},
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "numeric_columns": numeric_df.columns.tolist(),
            "sample_data": df.head(3).to_dict()
        }

        # Add correlation matrix for numeric columns
        if len(numeric_df.columns) > 1:
            profile["correlations"] = numeric_df.corr().to_dict()

        return profile

    def _detect_data_types(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Advanced column type detection with metadata."""
        column_info = {}

        for column in df.columns:
            info = {
                "unique_count": df[column].nunique(),
                "missing_count": df[column].isnull().sum(),
                "sample_values": df[column].dropna().head(3).tolist()
            }

            # Detect basic type
            if pd.api.types.is_numeric_dtype(df[column]):
                if df[column].nunique() < 10:
                    info["type"] = "categorical_numeric"
                else:
                    info["type"] = "continuous_numeric"
                    info["statistics"] = {
                        "mean": df[column].mean(),
                        "std": df[column].std(),
                        "skew": df[column].skew()
                    }
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                info["type"] = "datetime"
                info["range"] = {
                    "start": df[column].min().strftime("%Y-%m-%d"),
                    "end": df[column].max().strftime("%Y-%m-%d")
                }
            else:
                if df[column].nunique() < df.shape[0] * 0.05:
                    info["type"] = "categorical"
                    info["value_counts"] = df[column].value_counts().head(5).to_dict()
                else:
                    info["type"] = "text"
                    info["avg_length"] = df[column].str.len().mean()

            column_info[column] = info

        return column_info

    def _convert_to_serializable(self, obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(obj, dict):
        return {key: self._convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [self._convert_to_serializable(item) for item in obj]
    return obj

def _detect_data_types(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Advanced column type detection with metadata."""
    column_info = {}
    
    for column in df.columns:
        info = {
            "unique_count": int(df[column].nunique()),  # Convert to regular int
            "missing_count": int(df[column].isnull().sum()),  # Convert to regular int
            "sample_values": self._convert_to_serializable(df[column].dropna().head(3).tolist())
        }
        
        # Detect basic type
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].nunique() < 10:
                info["type"] = "categorical_numeric"
            else:
                info["type"] = "continuous_numeric"
                info["statistics"] = {
                    "mean": float(df[column].mean()),  # Convert to regular float
                    "std": float(df[column].std()),
                    "skew": float(df[column].skew())
                }
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            info["type"] = "datetime"
            info["range"] = {
                "start": df[column].min().strftime("%Y-%m-%d"),
                "end": df[column].max().strftime("%Y-%m-%d")
            }
        else:
            if df[column].nunique() < df.shape[0] * 0.05:
                info["type"] = "categorical"
                info["value_counts"] = {
                    str(k): int(v) 
                    for k, v in df[column].value_counts().head(5).items()
                }
            else:
                info["type"] = "text"
                info["avg_length"] = float(df[column].str.len().mean())
                
        column_info[column] = info
        
    return column_info

def _get_visualization_suggestions(self, column_info: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get LLM suggestions for visualizations."""
    # Convert column_info to JSON-serializable format
    serializable_info = self._convert_to_serializable(column_info)
    
    prompt = f"""
    Given these columns and their types:
    {json.dumps(serializable_info, indent=2)}
    
    Suggest 3 meaningful visualizations that would best represent the data.
    For each visualization, specify:
    1. The type of plot
    2. Which columns to use
    3. Why this visualization would be useful
    
    Return your response as a JSON array with objects containing:
    - plot_type: string (e.g., "scatter", "bar", "line", "heatmap")
    - columns: array of column names
    - description: string explaining why this visualization is useful
    """
    
    messages = [
        {"role": "system", "content": "You are a data visualization expert."},
        {"role": "user", "content": prompt}
    ]
    
    response = self._ask_llm(messages)
    try:
        return json.loads(response) if response else []
    except:
        return []

def _generate_story(self, analysis_results: List[str], profile: Dict[str, Any]) -> str:
    """Generate a narrative story from the analysis."""
    # Convert profile to JSON-serializable format
    serializable_profile = self._convert_to_serializable(profile)
    
    prompt = f"""
    Create an engaging data story based on these analysis results:
    
    Analysis Findings:
    {json.dumps(analysis_results, indent=2)}
    
    Dataset Profile:
    {json.dumps(serializable_profile, indent=2)}
    
    Write a compelling narrative that:
    1. Tells a story about what the data reveals
    2. Highlights key insights and their implications
    3. Makes the technical findings accessible to non-technical readers
    4. Includes section headings and proper markdown formatting
    5. Integrates references to the visualizations
    
    Format the story in markdown with clear sections.
    """
    
    messages = [
        {"role": "system", "content": "You are a data storyteller who transforms technical findings into engaging narratives."},
        {"role": "user", "content": prompt}
    ]
    
    return self._ask_llm(messages)
    def _get_llm_analysis(self, df: pd.DataFrame, profile: Dict[str, Any]) -> str:
        """Get LLM's analysis of the dataset."""
        prompt = f"""
        Analyze this dataset and provide insights:

        Dataset Information:
        - Shape: {profile['shape']}
        - Columns: {profile['columns']}
        - Data Types: {profile['dtypes']}
        - Missing Values: {profile['missing_values']}

        Numeric Summary: {profile['numeric_summary']}
        Categorical Columns: {profile['categorical_columns']}

        Please provide:
        1. Key insights about the data
        2. Potential patterns or trends
        3. Recommendations for further analysis
        4. Business implications (if applicable)
        5. Data quality issues and suggestions

        Format your response in markdown.
        """

        messages = [
            {"role": "system", "content": "You are a data scientist expert in analyzing datasets and finding meaningful insights."},
            {"role": "user", "content": prompt}
        ]

        return self._ask_llm(messages)

    def _get_visualization_suggestions(self, column_info: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get LLM suggestions for visualizations."""
        prompt = f"""
        Given these columns and their types:
        {json.dumps(column_info, indent=2)}

        Suggest 3 meaningful visualizations that would best represent the data.
        For each visualization, specify:
        1. The type of plot
        2. Which columns to use
        3. Why this visualization would be useful

        Return your response as a JSON array with objects containing:
        - plot_type: string (e.g., "scatter", "bar", "line", "heatmap")
        - columns: array of column names
        - description: string explaining why this visualization is useful
        """

        messages = [
            {"role": "system", "content": "You are a data visualization expert."},
            {"role": "user", "content": prompt}
        ]

        response = self._ask_llm(messages)
        try:
            return json.loads(response) if response else []
        except:
            return []

    def _generate_story(self, analysis_results: List[str], profile: Dict[str, Any]) -> str:
        """Generate a narrative story from the analysis."""
        prompt = f"""
        Create an engaging data story based on these analysis results:

        Analysis Findings:
        {json.dumps(analysis_results, indent=2)}

        Dataset Profile:
        {json.dumps(profile, indent=2)}

        Write a compelling narrative that:
        1. Tells a story about what the data reveals
        2. Highlights key insights and their implications
        3. Makes the technical findings accessible to non-technical readers
        4. Includes section headings and proper markdown formatting
        5. Integrates references to the visualizations

        Format the story in markdown with clear sections.
        """

        messages = [
            {"role": "system", "content": "You are a data storyteller who transforms technical findings into engaging narratives."},
            {"role": "user", "content": prompt}
        ]

        return self._ask_llm(messages)

    def analyze_dataset(self, file_path: str):
        """Main analysis method."""
        try:
            # Load and validate data
            df = self._validate_csv(file_path)
            profile = self._get_data_profile(df)
            column_info = self._detect_data_types(df)

            # Get LLM analysis
            llm_analysis = self._get_llm_analysis(df, profile)
            if llm_analysis:
                self.insights.append(llm_analysis)

            # Get visualization suggestions
            viz_suggestions = self._get_visualization_suggestions(column_info)
            for viz in viz_suggestions:
                self._create_visualization(df, viz)

            # Generate story
            story = self._generate_story(self.insights, profile)

            # Generate and save markdown report
            markdown = "# Dataset Analysis Report\n\n"
            markdown += story if story else "No story generated."

            with open("README.md", "w") as f:
                f.write(markdown)

        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            traceback.print_exc()

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    token = os.getenv("AIPROXY_TOKEN")

    if not token:
        print("Error: AIPROXY_TOKEN environment variable not set")
        sys.exit(1)

    try:
        analyzer = DatasetAnalyzer(token)
        analyzer.analyze_dataset(file_path)
        print("Analysis complete. Results saved in README.md")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()



