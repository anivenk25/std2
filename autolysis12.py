# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "requests>=2.28.0",
#    "pandas>=1.5.0",
#    "matplotlib>=3.5.0",
#    "seaborn>=0.12.0",
#    "numpy>=1.21.0",
#    "rich>=12.0.0",
#    "scipy>=1.9.0",
#    "scikit-learn>=1.0.0",
# ]
# description = "A script for data analysis and visualization."
# entry-point = "autolysis.py"
# ///

import os
import requests
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
import re
from pathlib import Path
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
    """Decorator to measure execution time of functions."""
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

class APIClient:
    """Handles API communication with LLM service."""
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
        """Make API request with error handling."""
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
    """Collection of statistical analysis methods."""
    
    @staticmethod
    def basic_stats(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for the DataFrame."""
        return {
            'summary': data.describe(),
            'missing': data.isnull().sum(),
            'dtypes': data.dtypes
        }

    @staticmethod
    def normality_test(data: pd.Series) -> Dict[str, Any]:
        """Perform a normality test on the data."""
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
        """Detect outliers in the data using IQR."""
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

class VisualizationStrategy(ABC):
    """Abstract base class for visualization strategies."""
    @abstractmethod
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        """Create a visualization and save it to a file."""
        pass

class CorrelationHeatmap(VisualizationStrategy):
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        """Create a correlation heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(f"Correlation Heatmap - {title}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

class DistributionPlot(VisualizationStrategy):
    def create_visualization(self, df: pd.DataFrame, fig_path: Path, title: str) -> None:
        """Create distribution plots for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        if n_cols == 0:
            return
            
        plt.figure(figsize=(15, 5 * ((n_cols + 1) // 2)))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(((n_cols + 1) // 2), 2, i)
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
        
        plt.suptitle(f"Distribution Analysis - {title}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

class LLMAnalyzer:
    """Class to analyze datasets using LLM for code generation and insights."""
    def __init__(self):
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise EnvironmentError("AIPROXY_TOKEN is not set. Please set it as an environment variable.")

        self.proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.figure_counter = 0

    def _save_and_close_plot(self, title: str):
        """Save the current plot to a file and close it."""
        self.figure_counter += 1
        filename = f'plot_{self.figure_counter}.png'
        plt.title(title)
        plt.savefig(filename)
        print(f"Plot saved as: {filename}")
        plt.close()

    def _make_llm_request(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make a request to the LLM API with error handling."""
        try:
            data = {
                "model": "gpt-4o-mini",
                "messages": messages
            }
            response = requests.post(self.proxy_url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing API response: {str(e)}")
            return None

    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract Python code blocks from markdown-formatted text."""
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        return code_blocks if code_blocks else []

    def _execute_code_safely(self, code: str, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Execute code with safety measures and return success status and error message."""
        try:
            # Ensure the code uses the provided DataFrame
            if 'pd.read_csv' in code:
                raise ValueError("Code should not read from CSV files directly. Use the provided DataFrame 'df'.")

            # Modify the code to save plots instead of showing them
            code = code.replace('plt.show()', 'self._save_and_close_plot("Generated Plot")')

            # Create a restricted locals dictionary with only necessary objects
            local_dict = {
                'pd': pd, 
                'plt': plt, 
                'sns': sns, 
                'df': df, 
                'analyzer': self
            }

            # Execute the code in the restricted environment
            exec(code, {'__builtins__': __builtins__}, local_dict)

            return True, None
        except Exception as e:
            error_msg = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            return False, error_msg

    def _fix_code_recursively(self, code: str, error_msg: str, df: pd.DataFrame, max_attempts: int = 3) -> bool:
        """Recursively try to fix code using LLM until it works or max attempts reached."""
        attempt = 0
        while attempt < max_attempts:
            fix_prompt = f"""
            The following Python code generated an error:
            ```python
            {code}
            ```

            Error message:
            {error_msg}

            Please provide a fixed version of the code that:
            1. Handles the error properly
            2. Uses only pandas, matplotlib.pyplot, and seaborn
            3. Works with the DataFrame that has these columns: {df.columns.tolist()}
            4. Includes proper error handling
            5. Uses plt.figure() before creating each plot
            6. Uses plt.show() after each plot is complete
            7. Does not reference any specific CSV files

            Provide ONLY the corrected code block, no explanations.
            """

            messages = [
                {"role": "system", "content": "You are a Python expert focused on data analysis and visualization."},
                {"role": "user", "content": fix_prompt}
            ]

            fixed_content = self._make_llm_request(messages)
            if not fixed_content:
                return False

            fixed_code_blocks = self._extract_code_blocks(fixed_content)
            if not fixed_code_blocks:
                fixed_code = fixed_content  # If no code blocks found, try using the entire response
            else:
                fixed_code = fixed_code_blocks[0]

            success, new_error = self._execute_code_safely(fixed_code, df)
            if success:
                return True

            error_msg = new_error
            attempt += 1

        return False

    def analyze_dataset(self, file_path: str):
        """Main method to analyze the dataset."""
        try:
            # Validate file path
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"The file '{file_path}' does not exist.")
            if not path.is_file():
                raise ValueError(f"'{file_path}' is not a file.")
            if path.suffix.lower() != '.csv':
                raise ValueError(f"'{file_path}' is not a CSV file.")

            print(f"Loading dataset from: {file_path}")

            # Load and validate dataset with error handling for encoding
            try:
                df = pd.read_csv(file_path, encoding='utf-8')  # Try UTF-8 first
            except UnicodeDecodeError:
                print("UTF-8 decoding failed, trying ISO-8859-1 encoding...")
                df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Fallback to ISO-8859-1
            except pd.errors.EmptyDataError:
                raise ValueError("The CSV file is empty.")
            except pd.errors.ParserError:
                raise ValueError("Error parsing the CSV file. Please ensure it's properly formatted.")

            if df.empty:
                raise ValueError("Dataset is empty")

            print(f"Successfully loaded dataset with shape: {df.shape}")

            # Generate initial data description
            data_description = (
                f"Dataset Overview:\n"
                f"Columns: {df.columns.tolist()}\n"
                f"Shape: {df.shape}\n"
                f"Sample data:\n{df.head(3).to_string()}\n"
                f"Data types:\n{df.dtypes.to_string()}"
            )

            print("\nGenerating analysis...")

            # Get initial analysis suggestions
            initial_prompt = f"""
            Given this dataset description:
            {data_description}

            Generate Python code that:
            1. Creates meaningful visualizations using matplotlib and seaborn
            2. Calculates relevant summary statistics
            3. Identifies key patterns or relationships
            4. Handles potential errors (missing values, invalid data)
            5. Uses plt.figure() before creating each plot
            6. Uses plt.show() after each plot is complete
            7. Does not reference any specific CSV files

            Provide the code in a Python code block.
            """

            messages = [
                {"role": "system", "content": "You are a data scientist specialized in exploratory data analysis."},
                {"role": "user", "content": initial_prompt}
            ]

            # Get and execute initial analysis
            analysis_content = self._make_llm_request(messages)
            if analysis_content:
                code_blocks = self._extract_code_blocks(analysis_content)
                for code in code_blocks:
                    success, error_msg = self._execute_code_safely(code, df)
                    if not success:
                        print(f"Initial code execution failed. Attempting to fix...")
                        if not self._fix_code_recursively(code, error_msg, df):
                            print("Failed to fix code after maximum attempts")

            # Generate final insights
            insights = self._generate_final_insights(df)

            # Generate the epic story
            print("\nGenerating the epic story...")
            self._generate_epic_story(df, insights)

        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            traceback.print_exc()

    def _generate_final_insights(self, df: pd.DataFrame):
        """Generate final insights after analysis."""
        # Create a summary of the numerical analysis
        numerical_summary = df.describe().to_string()

        # Create a summary of missing values
        missing_values = df.isnull().sum().to_string()

        insight_prompt = f"""
        Analyze this dataset based on the following information:

        1. Dataset Statistics:
        {numerical_summary}

        2. Missing Values Analysis:
        {missing_values}

        3. Generated Visualizations:
        - {self.figure_counter} plots were generated analyzing different aspects of the data

        Please provide:
        1. Key patterns and trends from the data
        2. Important statistical findings
        3. Notable relationships between variables
        4. Insights about the distribution of ratings
        5. Any interesting observations about the dataset and their characteristics
        6. Recommendations for stakeholders

        Format the response with clear headers and bullet points.
        """

        messages = [
            {"role": "system", "content": "You are a data analyst specializing in dataset analysis."},
            {"role": "user", "content": insight_prompt}
        ]

        insights = self._make_llm_request(messages)
        if insights:
            print("\nKey Insights:")
            print(insights)
            return str(insights)

    def _generate_epic_story(self, df: pd.DataFrame, insights: str):
        """Generate an epic narrative based on the data analysis."""
        # Create summaries for context
        numerical_summary = df.describe().to_string()
        missing_values = df.isnull().sum().to_string()
        
        # Automatically determine the subject and genre
        subject = self._determine_subject(df)
        genre = self._determine_genre(df)
        
        story_prompt = f"""
        In the vibrant world of data, where every number tells a story and every insight sparks a connection, you are the beloved storyteller, a modern bard navigating the complexities of {subject} through this dataset.

        - {df.shape} data points each representing a unique journey of {subject}.
        - {df.columns.tolist()}, each column a chapter in the saga of {subject}.
        -  {df.head(3).to_string()}, where the first sparks of {subject} intertwine.

        - {missing_values}, like unspoken words in a love letter, leaving gaps in the narrative that yearn to be filled with understanding.

        - {self.figure_counter} enchanting illustrations conjured from the depths of analysis, each revealing a facet of {subject}.

        **Final Insights:**
        - {insights}  
        Craft a heartwarming narrative that unfolds like a contemporary {genre}, filled with emotional growth and profound insights (IMPORTANT : REFER AND USE THE FINAL INSIGHTS SECTION THROUGHT THE STORY AND MAKE SURE THAT THE STORY IS CONSISTENT WITH THEM also for every claim made weave in the numbers too also make the process of coming to every conclusion sumer dramatic)

        MAKE THE PROCESS OF ARRIVING TO THESE CONCLUSIONS VERY GRIPPING AND UNIQUE 

        TUG ON EMOTIONS 

        ADD DRAMA ADD LOVE ADD THRILL ADD HERO ENTRY AND COOL SHIT LIKE THAT 

        THE STORY MUST BE VERY MEMEORABLE AND MUST APPEASE INDIAN AUDIENCE BUT YOU CAN MAKE THE STORY NON INDIAN TOO IF NEEDED.

        AGAIN THE FINAL INSIGHTS SECTION IS GODLIKE -- FOLLOW IT AND PRESENT AS MUCH INFO FROM THAT IN THE STORY AS POSSIBLE

        USE {insights}
        """

        messages = [
            {"role": "system", "content": "You are an immortal storyteller who transforms data into legendary tales."},
            {"role": "user", "content": story_prompt}
        ]
        
        story = self._make_llm_request(messages)
        if story:
            print("\n" + "="*50)
            print("The Legend of the Literary Realms")
            print("="*50 + "\n")
            print(story)

    def _determine_subject(self, df: pd.DataFrame) -> str:
        """Determine the subject of the dataset based on its content."""
        # Example heuristic: use the most common word in the first few rows of the DataFrame
        text_data = ' '.join(df.astype(str).values.flatten())
        words = re.findall(r'\w+', text_data.lower())
        most_common_word = pd.Series(words).value_counts().idxmax()
        return most_common_word.capitalize()

    def _determine_genre(self, df: pd.DataFrame) -> str:
        """Determine the genre of the dataset based on its content."""
        # Example heuristic: analyze column names for genre-related keywords
        genre_keywords = ['rating', 'review', 'score', 'feedback', 'sentiment']
        for column in df.columns:
            if any(keyword in column.lower() for keyword in genre_keywords):
                return "Analysis of Ratings"  # Example genre
        return "Various Themes"  # Default genre if no keywords found

class DataAnalyzer:
    """Enhanced data analyzer with comprehensive analysis capabilities."""
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.api_client = APIClient()
        self.stats_analyzer = StatisticalMethods()
        self.llm_analyzer = LLMAnalyzer()  # Integrate LLMAnalyzer
        self.visualization_strategies = [
            CorrelationHeatmap(),
            DistributionPlot()
        ]
        self.plots: List[str] = []
        
    @timeit
    def analyze_dataset(self, file_path: str):
        """Main method to analyze the dataset."""
        try:
            start_time = time.time()
            self._create_output_directory()
            df = self._load_and_validate_dataset(file_path)
            print(f"Successfully loaded dataset with shape: {df.shape}")

            # Use LLMAnalyzer for analysis
            self.llm_analyzer.analyze_dataset(file_path)

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
        """Create output directory for saving results."""
        if self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True)

    def _load_and_validate_dataset(self, file_path: str) -> pd.DataFrame:
        """Load and validate the dataset from the given file path."""
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
        """Generate visualizations for the dataset."""
        for i, strategy in enumerate(self.visualization_strategies):
            viz_path = self.config.output_dir / f'visualization_{i}.png'
            strategy.create_visualization(df, viz_path, f"Analysis {i+1}")
            self.plots.append(viz_path.name)

    def _generate_insights(self, df: pd.DataFrame, stats: Dict[str, Any]) -> str:
        """Generate insights based on the dataset and statistical analysis."""
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
        """Generate a narrative based on the dataset analysis."""
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
        """Determine the subject of the dataset based on its content."""
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
        """Generate a README file with the analysis narrative and visualizations."""
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
    # Ensure the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    config = AnalysisConfig(output_dir=Path(Path(file_path).stem))
    analyzer = DataAnalyzer(config)
    analyzer.analyze_dataset(file_path)

if __name__ == "__main__":
    main()
