import os
import requests
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import traceback
import re
from pathlib import Path
import sys
import shutil
from autoviz.AutoViz_Class import AutoViz_Class  # Import AutoViz

class LLMAnalyzer:
    def __init__(self, output_dir: Path):
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise EnvironmentError("AIPROXY_TOKEN is not set. Please set it as an environment variable.")

        self.proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.plots = []  # Store plot filenames
        self.output_dir = output_dir  # Store output directory path

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

    def analyze_dataset(self, file_path: str):
        """Main method to analyze the dataset."""
        try:
            # Create output directory if it doesn't exist
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)

            # Validate file path
            path = Path(file_path)
            if not path.exists() or not path.is_file() or path.suffix.lower() != '.csv':
                raise ValueError(f"Invalid file: {file_path}")

            print(f"Loading dataset from: {file_path}")

            # Load dataset with error handling for encoding
            df = self._load_dataset(file_path)

            if df.empty:
                raise ValueError("Dataset is empty")

            print(f"Successfully loaded dataset with shape: {df.shape}")

            # Use AutoViz for EDA
            av = AutoViz_Class()
            dfte = av.AutoViz(file_path, sep=',', depVar='', dfte=None, header=0, verbose=1, lowess=False,
                              chart_format='svg', max_rows_analyzed=150000, max_cols_analyzed=30,
                              save_plot_dir=str(self.output_dir))  # Save plots to output directory

            # Check if AutoViz returned None
            if dfte is None:
                print("AutoViz failed, falling back to LLM for EDA.")
                insights = self._generate_llm_insights(df)
            else:
                # Retrieve insights from AutoViz
                insights = self._generate_final_insights(dfte)

            # Generate the epic story with integrated insights
            print("\nGenerating the epic story...")
            story = self._generate_epic_story(dfte if dfte is not None else df, insights)

            # Generate README.md
            if story:
                self.generate_readme(story)

        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            traceback.print_exc()

    def _load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset with error handling for encoding."""
        try:
            return pd.read_csv(file_path, encoding='utf-8')  # Try UTF-8 first
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying ISO-8859-1 encoding...")
            return pd.read_csv(file_path, encoding='ISO-8859-1')  # Fallback to ISO-8859-1
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except pd.errors.ParserError:
            raise ValueError("Error parsing the CSV file. Please ensure it's properly formatted.")

    def _generate_final_insights(self, df: pd.DataFrame) -> str:
        """Generate final insights after analysis using AutoViz."""
        subject = self._determine_subject(df)
        numerical_summary = df.describe().to_string()
        missing_values = df.isnull().sum().to_string()

        auto_viz_insights = f"""
        **AutoViz Insights:**
        - Dataset Statistics:\n{numerical_summary}
        - Missing Values Analysis:\n{missing_values}
        """

        insight_prompt = f"""
        Analyze this {subject} dataset based on the following information:

        {auto_viz_insights}

        Please provide:
        1. Key patterns and trends from the data
        2. Important statistical findings
        3. Notable relationships between variables
        4. Insights about the distribution of ratings
        5. Any interesting observations about {subject} and their characteristics
        6. Recommendations for stakeholders

        Format the response with clear headers and bullet points.
        """

        messages = [
            {"role": "system", "content": "You are a data analyst specializing in book data and user ratings analysis."},
            {"role": "user", "content": insight_prompt}
        ]

        insights = self._make_llm_request(messages)
        if insights:
            print("\nKey Insights:")
            print(insights)
            return str(insights)

    def _generate_epic_story(self, df: pd.DataFrame, insights: str) -> str:
        """Generate an epic narrative based on the data analysis."""
        subject = self._determine_subject(df)
        genre = self._determine_genre(df)

        story_prompt = f"""
        In the vibrant world of data, where every number tells a story and every insight sparks a connection, you are the beloved storyteller, a modern bard navigating the complexities of {subject} through this dataset.

        - {df.shape} data points each representing a unique journey of {subject}.
        - {df.columns.tolist()}, each column a chapter in the saga of {subject}.
        - {df.head(3).to_string()}, where the first sparks of {subject} intertwine.

        - {insights}

        Craft a heartwarming narrative that unfolds like a contemporary {genre}, filled with emotional growth and profound insights.
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
            return story

    def generate_readme(self, story: str):
        """Generate README.md with the story and embedded plots."""
        readme_content = "# Data Analysis Story\n\n"
        readme_content += story + "\n\n"

        # Add plots section
        readme_content += "## Supporting Visualizations\n\n"
        for plot in self.plots:
            readme_content += f"![{plot}]({plot})\n\n"

        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"README.md generated at: {readme_path}")

    def _determine_subject(self, df: pd.DataFrame) -> str:
        """Determine the subject of the dataset using LLM analysis."""
        data_description = (
            f"Dataset Overview:\n"
            f"Columns: {df.columns.tolist()}\n"
            f"Shape: {df.shape}\n"
            f"Sample data:\n{df.head(3).to_string()}\n"
            f"Data types:\n{df.dtypes.to_string()}"
        )

        subject_prompt = f"""
        Analyze the following dataset and identify its primary subject matter:

        {data_description}

        Based on the columns, data types, and sample data, determine the most likely subject of this dataset. 
        Provide a concise, single-word or short-phrase subject that best represents the core focus of the data.

        Respond with ONLY the subject, without any additional explanation or context.
        """

        messages = [
            {"role": "system", "content": "You are an expert data analyst skilled at quickly identifying dataset subjects."},
            {"role": "user", "content": subject_prompt}
        ]

        subject = self._make_llm_request(messages)
        
        # Fallback if LLM fails
        if not subject or len(subject.split()) > 3:
            text_data = ' '.join(df.astype(str).values.flatten())
            words = re.findall(r'\w+', text_data.lower())
            subject = pd.Series(words).value_counts().idxmax()
        
        return subject.capitalize().strip()

    def _determine_genre(self, df: pd.DataFrame) -> str:
        """Determine the genre of the dataset using LLM analysis."""
        data_description = (
            f"Dataset Overview:\n"
            f"Columns: {df.columns.tolist()}\n"
            f"Shape: {df.shape}\n"
            f"Sample data:\n{df.head(3).to_string()}\n"
            f"Data types:\n{df.dtypes.to_string()}"
        )

        genre_prompt = f"""
        Analyze the following dataset and identify its genre or primary analytical category:

        {data_description}

        Based on the columns, data types, sample data, and overall structure, 
        determine the most appropriate genre for a story based on this dataset. 
        Provide a concise genre that best captures the essence of the dataset.
        Respond with ONLY the genre, without any additional explanation or context.
        """

        messages = [
            {"role": "system", "content": "You are an expert data analyst skilled at categorizing datasets."},
            {"role": "user", "content": genre_prompt}
        ]

        genre = self._make_llm_request(messages)
        
        # Fallback if LLM fails
        if not genre or len(genre.split()) > 4:
            genre_keywords = ['rating', 'review', 'score', 'feedback', 'sentiment']
            for column in df.columns:
                if any(keyword in column.lower() for keyword in genre_keywords):
                    genre = "Analysis of Ratings"
                    break
            else:
                genre = "Various Themes"
        
        return genre.strip()

def main():
    """Main function to run the analysis."""
    try:
        if len(sys.argv) != 2:
            print("Usage: uv run autolysis.py dataset.csv")
            sys.exit(1)

        file_path = Path(sys.argv[1])
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Create output directory named after the CSV file (without extension)
        output_dir = Path(file_path.stem)
        if output_dir.exists():
            # If directory exists, remove it and its contents
            shutil.rmtree(output_dir)

        # Initialize and run analyzer with output directory
        analyzer = LLMAnalyzer(output_dir)
        analyzer.analyze_dataset(str(file_path))

    except Exception as e:
        print(f"Program failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
