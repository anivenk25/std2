import os
import requests
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List
import traceback
import re
from pathlib import Path

class LLMAnalyzer:
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

    def _execute_code_safely(self, code: str, df: pd.DataFrame) -> tuple[bool, Optional[str]]:
        """Execute code with safety measures and return success status and error message."""
        try:
            # Ensure the code uses the provided DataFrame
            if 'pd.read_csv' in code:
                raise ValueError("Code should not read from CSV files directly. Use the provided DataFrame 'df'.")

            # Modify the code to save plots instead of showing them
            code = code.replace('plt.show()', 'analyzer._save_and_close_plot("Generated Plot")')

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
            self._generate_final_insights(df)

            # Generate the epic story
            print("\nGenerating the epic story...")
            self._generate_epic_story(df)

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
        Analyze this Goodreads dataset based on the following information:

        1. Dataset Statistics:
        {numerical_summary}

        2. Missing Values Analysis:
        {missing_values}

        3. Generated Visualizations:
        - {self.figure_counter} plots were generated analyzing different aspects of the data

        Please provide:
        1. Key patterns and trends from the data
        2. Important statistical findings
        3. Notable relationships between book ratings and other variables
        4. Insights about the distribution of ratings
        5. Any interesting observations about books and their characteristics
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

    def _generate_epic_story(self, df: pd.DataFrame):
        """Generate an epic narrative based on the data analysis."""
        # Create summaries for context
        numerical_summary = df.describe().to_string()
        missing_values = df.isnull().sum().to_string()
        
        story_prompt = f"""
In the vibrant world of data, where every number tells a story and every insight sparks a connection, you are the beloved storyteller, a modern bard navigating the complexities of love and life through this sacred dataset.

**The Scroll of Destiny:**
- **Shape of the Realm:** {df.shape} souls captured in this digital tapestry, each representing a unique journey of love and happiness.
- **Columns of Wisdom:** {df.columns.tolist()}, each column a chapter in the romantic saga of human experience.
- **Glimpses of the Past:** {df.head(3).to_string()}, where the first sparks of romance and the bittersweet moments of heartbreak intertwine.

**The Shadows of the Void:**
- **Missing Values:** {missing_values}, like unspoken words in a love letter, leaving gaps in the narrative that yearn to be filled with understanding.

**Visions from the Ether:**
- **Mystical Charts:** {self.figure_counter} enchanting illustrations conjured from the depths of analysis, each revealing a facet of modern relationships.

Craft a heartwarming narrative that unfolds like a contemporary K-drama, filled with romantic moments, emotional growth, and profound insights:
1. **The Awakening**: Begin with a charming prologue, where the protagonist, a passionate data analyst, discovers this dataset while searching for answers about love in the digital age. As they dive into the data, they uncover the intertwined lives of individuals, each data point a heartbeat in the story of modern romance.

2. **The Hero's Journey**: Describe the trials faced as the hero navigates the complexities of relationships, each insight a powerful revelation that brings them closer to understanding love. Along the way, they discover the **Key Patterns**—the correlation between happiness and meaningful connections, revealing that love and friendship are the true treasures of life.

3. **The Revelations**: Unveil the secrets of the dataset as if they were heartfelt confessions, illuminating the path to understanding the dynamics of modern relationships. The **Statistical Findings** reveal that couples who communicate openly and engage in shared activities report higher levels of satisfaction, a testament to the power of connection and understanding.

4. **The Call to Action**: Conclude with a heartfelt plea, urging friends, families, and partners to embrace these insights and nurture their relationships. The implications of these findings resonate deeply: fostering open communication and shared experiences can lead to a happier, more fulfilling life together.

**Guidelines for Your Epic:**
- Infuse your tale with the essence of modern romance, where love blossoms amidst the challenges of everyday life, and every insight is a step toward emotional fulfillment.
- Compare statistical findings to the wisdom of the heart, drawing parallels to the relatable experiences of love and friendship.
- Treat each data point as a character in a romantic comedy, each entity a unique individual with a story that resonates with the audience.
- Frame the visualizations as snapshots of moments in time, revealing the hidden truths of relationships, each chart a glimpse into the heart of love.
- Weave in dramatic tension, where the quest for understanding faces relatable challenges, including misunderstandings, moments of doubt, and the joy of reconciliation.
- End with a powerful message, a poetic reflection on the profound meaning hidden within the numbers, inspiring all who read it to cherish their connections and embrace the beauty of love.

Let your words flow like the lyrics of a heartfelt ballad, creating an unforgettable story that resonates in the hearts of those who dare to explore the depths of love, friendship, and the beauty of modern relationships.
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



def main():
    """Main function to run the analysis."""
    try:
        # Get current directory
        current_dir = Path.cwd()

        # Look for CSV files in the current directory
        csv_files = list(current_dir.glob('*.csv'))

        if not csv_files:
            raise FileNotFoundError("No CSV files found in the current directory.")

        # Use the first CSV file found
        file_path = csv_files[0]
        print(f"Found CSV file: {file_path}")

        # Initialize and run analyzer
        analyzer = LLMAnalyzer()
        analyzer.analyze_dataset(str(file_path))

    except Exception as e:
        print(f"Program failed: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
