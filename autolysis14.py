import os
import requests
import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple
import traceback
import re
import chardet
from pathlib import Path
import sys
from dataclasses import dataclass
from enum import Enum, auto

class StatTestType(Enum):
    TTEST = auto()
    ANOVA = auto()
    CHI_SQUARE = auto()
    CORRELATION = auto()
    REGRESSION = auto()
    NORMALITY = auto()
    MANN_WHITNEY = auto()

@dataclass
class StatTestResult:
    test_type: StatTestType
    statistic: float
    p_value: float
    interpretation: str
    additional_info: Dict[str, Any] = None

class StatisticalAnalyzer:
    """Handles statistical analysis operations."""
    
    # Define a constant for significance level
    SIGNIFICANCE_LEVEL = 0.05

    @staticmethod
    def run_statistical_test(test_type: StatTestType, data: pd.DataFrame, 
                           columns: List[str], **kwargs) -> StatTestResult:
        """Run specified statistical test and return results."""
        try:
            # Check if columns are present in the DataFrame
            if not all(col in data.columns for col in columns):
                raise ValueError("One or more columns are not present in the DataFrame.")
            
            if test_type == StatTestType.TTEST:
                return StatisticalAnalyzer._run_ttest(data, columns, **kwargs)
            elif test_type == StatTestType.CORRELATION:
                return StatisticalAnalyzer._run_correlation(data, columns)
            elif test_type == StatTestType.NORMALITY:
                return StatisticalAnalyzer._run_normality_test(data, columns[0])
            elif test_type == StatTestType.CHI_SQUARE:
                return StatisticalAnalyzer._run_chi_square(data, columns)
            elif test_type == StatTestType.MANN_WHITNEY:
                return StatisticalAnalyzer._run_mann_whitney(data, columns, **kwargs)
            # TODO: Implement ANOVA and REGRESSION tests
        except Exception as e:
            return StatTestResult(
                test_type=test_type,
                statistic=float('nan'),
                p_value=float('nan'),
                interpretation=f"Test failed: {str(e)}"
            )

    @staticmethod
    def _run_ttest(data: pd.DataFrame, columns: List[str], 
                   paired: bool = False) -> StatTestResult:
        """Perform t-test between two columns."""
        if len(columns) != 2:
            raise ValueError("T-test requires exactly two columns")
            
        col1, col2 = columns
        if paired:
            stat, p_value = stats.ttest_rel(data[col1], data[col2])
        else:
            stat, p_value = stats.ttest_ind(data[col1], data[col2])
            
        interpretation = (f"{'Paired' if paired else 'Independent'} t-test result: "
                        f"{'Significant' if p_value < StatisticalAnalyzer.SIGNIFICANCE_LEVEL else 'Not significant'} "
                        f"difference between {col1} and {col2}")
        
        return StatTestResult(StatTestType.TTEST, stat, p_value, interpretation)

    @staticmethod
    def _run_correlation(data: pd.DataFrame, columns: List[str]) -> StatTestResult:
        """Perform correlation analysis between columns."""
        corr_matrix = data[columns].corr()
        
        # Get average correlation excluding self-correlations
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        avg_corr = corr_matrix.where(mask).mean().mean()
        
        interpretation = (f"Average correlation between variables: {avg_corr:.3f}\n"
                        f"Strongest correlation: {corr_matrix.unstack().sort_values()[-2]:.3f}")
        
        return StatTestResult(
            StatTestType.CORRELATION,
            avg_corr,
            1.0,  # p-value not applicable
            interpretation,
            {'correlation_matrix': corr_matrix}
        )

    @staticmethod
    def _run_normality_test(data: pd.DataFrame, column: str) -> StatTestResult:
        """Perform Shapiro-Wilk normality test."""
        stat, p_value = stats.shapiro(data[column])
        
        interpretation = (f"Normality test for {column}: "
                        f"{'Normal' if p_value > StatisticalAnalyzer.SIGNIFICANCE_LEVEL else 'Non-normal'} distribution")
        
        return StatTestResult(StatTestType.NORMALITY, stat, p_value, interpretation)

    @staticmethod
    def _run_chi_square(data: pd.DataFrame, columns: List[str]) -> StatTestResult:
        """Perform chi-square test of independence."""
        contingency = pd.crosstab(data[columns[0]], data[columns[1]])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        interpretation = (f"Chi-square test between {columns[0]} and {columns[1]}: "
                        f"{'Significant' if p_value < StatisticalAnalyzer.SIGNIFICANCE_LEVEL else 'No significant'} relationship")
        
        return StatTestResult(StatTestType.CHI_SQUARE, chi2, p_value, interpretation)

    @staticmethod
    def _run_mann_whitney(data: pd.DataFrame, columns: List[str], 
                         group_col: str = None) -> StatTestResult:
        """Perform Mann-Whitney U test."""
        if group_col:
            groups = data[group_col].unique()
            if len(groups) != 2:
                raise ValueError("Mann-Whitney test requires exactly two groups")
            
            group1 = data[data[group_col] == groups[0]][columns[0]]
            group2 = data[data[group_col] == groups[1]][columns[0]]
        else:
            if len(columns) != 2:
                raise ValueError("Mann-Whitney test requires exactly two columns")
            group1 = data[columns[0]]
            group2 = data[columns[1]]
            
        stat, p_value = stats.mannwhitneyu(group1, group2)
        
        interpretation = (f"Mann-Whitney U test result: "
                        f"{'Significant' if p_value < StatisticalAnalyzer.SIGNIFICANCE_LEVEL else 'No significant'} "
                        f"difference between groups")
        
        return StatTestResult(StatTestType.MANN_WHITNEY, stat, p_value, interpretation)

    @staticmethod
    def generate_plot(data: pd.DataFrame, columns: List[str], test_type: StatTestType):
        """Generate plots based on the statistical test results."""
        if test_type == StatTestType.CORRELATION:
            plt.figure(figsize=(10, 6))
            sns.heatmap(data[columns].corr(), annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.savefig('correlation_matrix.png')
            plt.close()
        elif test_type == StatTestType.NORMALITY:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[columns[0]], kde=True)
            plt.title(f'Distribution of {columns[0]}')
            plt.savefig(f'distribution_{columns[0]}.png')
            plt.close()
        elif test_type == StatTestType.CHI_SQUARE:
            contingency = pd.crosstab(data[columns[0]], data[columns[1]])
            sns.heatmap(contingency, annot=True, cmap='Blues')
            plt.title(f'Chi-Square Test: {columns[0]} vs {columns[1]}')
            plt.savefig(f'chi_square_{columns[0]}_{columns[1]}.png')
            plt.close()
        # Add more plots for other test types as needed

class LLMAnalyzer:
    def __init__(self):
        self.token = os.getenv("AIPROXY_TOKEN")
        if not self.token:
            raise EnvironmentError("AIPROXY_TOKEN is not set")

        self.proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.figure_counter = 0
        self.plots = []
        self.stat_analyzer = StatisticalAnalyzer()
        self.analysis_results = []

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

    def _send_plots_for_insights(self) -> str:
        """Send generated plots to the LLM and get insights."""
        insights = ""
        for plot in self.plots:
            messages = [
                {"role": "system", "content": "You are a data analyst specializing in interpreting visual data."},
                {"role": "user", "content": f"Please analyze the following plot and provide insights: {plot}"}
            ]
            plot_insight = self._make_llm_request(messages)
            if plot_insight:
                insights += f"\nInsights from {plot}:\n{plot_insight}\n"
        return insights

    def _request_statistical_tests(self, df: pd.DataFrame) -> List[Tuple[StatTestType, List[str]]]:
        """Request the LLM to suggest statistical tests based on the dataset."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        prompt = f"""
        Given the following dataset characteristics, suggest relevant statistical tests to perform within 2 minutes:
        
        Numeric Columns: {numeric_cols}
        Categorical Columns: {categorical_cols}
        
        Please respond in the following format:
        - TestType: [Column1, Column2]
        - TestType: [Column1]
        """

        messages = [
            {"role": "system", "content": "You are a data analyst who suggests statistical tests based on dataset characteristics."},
            {"role": "user", "content": prompt}
        ]

        response = self._make_llm_request(messages)
        return self._parse_llm_response(response)

    def _parse_llm_response(self, response: str) -> List[Tuple[StatTestType, List[str]]]:
        """Parse the LLM response to extract statistical tests and columns."""
        tests = []
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith('-'):
                parts = line[1:].strip().split(':')
                test_type_str = parts[0].strip()
                columns = eval(parts[1].strip())  # Convert string representation of list to actual list
                test_type = StatTestType[test_type_str.upper().replace(" ", "_")]
                tests.append((test_type, columns))
        return tests

    def _determine_statistical_tests(self, df: pd.DataFrame) -> List[Tuple[StatTestType, List[str]]]:
        """Determine appropriate statistical tests based on data characteristics."""
        # First, try to get suggestions from the LLM
        try:
            return self._request_statistical_tests(df)
        except Exception as e:
            print(f"Error requesting statistical tests: {str(e)}")
            return self._default_statistical_tests(df)

    def _default_statistical_tests(self, df: pd.DataFrame) -> List[Tuple[StatTestType, List[str]]]:
        """Fallback method to determine statistical tests based on the dataset."""
        tests = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Correlation analysis for numeric columns
        if len(numeric_cols) >= 2:
            tests.append((StatTestType.CORRELATION, numeric_cols))

        # Normality tests for important numeric columns
        for col in numeric_cols:  # Test all numeric columns
            tests.append((StatTestType.NORMALITY, [col]))

        # Chi-square tests for categorical columns
        if len(categorical_cols) >= 2:
            for i in range(len(categorical_cols) - 1):
                for j in range(i + 1, len(categorical_cols)):
                    tests.append((StatTestType.CHI_SQUARE, 
                                [categorical_cols[i], categorical_cols[j]]))

        # Regression analysis if we have potential target variables
        if len(numeric_cols) >= 3:
            tests.append((StatTestType.REGRESSION, numeric_cols))

        return tests

    def _run_statistical_analysis(self, df: pd.DataFrame) -> str:
        """Run statistical analysis and return formatted results."""
        tests = self._determine_statistical_tests(df)
        results = []

        for test_type, columns in tests:
            try:
                result = self.stat_analyzer.run_statistical_test(test_type, df, columns)
                self.stat_analyzer.generate_plot(df, columns, test_type)  # Generate plots
                self.plots.append(f"{test_type.name.lower()}_{columns[0]}_{columns[1] if len(columns) > 1 else ''}.png")  # Track plots
                self.analysis_results.append(result)
                results.append(f"\n{result.interpretation}")
            except Exception as e:
                print(f"Error running {test_type} test: {str(e)}")

        return "\n".join(results)

    def analyze_dataset(self, file_path: str):
        """Main method to analyze the dataset with enhanced statistical analysis."""
        try:
            # Validate the file path
            if not Path(file_path).is_file():
                raise FileNotFoundError(f"The file {file_path} does not exist.")

            # Detect encoding
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read())
                encoding = result['encoding']
                print(f"Detected encoding: {encoding}")

            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded dataset with shape: {df.shape}")

            # Run statistical analysis
            print("\nPerforming statistical analysis...")
            statistical_results = self._run_statistical_analysis(df)
            print("Statistical analysis completed.")

            # Generate insights from plots
            plot_insights = self._send_plots_for_insights()

            # Generate initial data description with statistical insights
            data_description = (
                f"Dataset Overview:\n"
                f"Columns: {df.columns.tolist()}\n"
                f"Shape: {df.shape}\n"
                f"Sample data:\n{df.head(3).to_string()}\n"
                f"Data types:\n{df.dtypes.to_string()}\n"
                f"\nStatistical Findings:\n{statistical_results}"
                f"\n\nPlot Insights:\n{plot_insights}"
            )

            # Generate insights
            insights = self._generate_final_insights(df)
            if insights:
                insights = self._enhance_insights_with_statistics(insights, self.analysis_results)

            # Include insights in the story
            story = self._generate_epic_story(df, insights, plot_insights)
            if story:
                self.generate_readme(story)

        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            traceback.print_exc()

    def _generate_final_insights(self, df: pd.DataFrame):
        """Generate final insights after analysis."""
        subject = self._determine_subject(df)
        # Create a summary of the numerical analysis
        numerical_summary = df.describe().to_string()

        # Create a summary of missing values
        missing_values = df.isnull().sum().to_string()

        insight_prompt = f"""
        Analyze this {subject} dataset based on the following information:

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

    def _generate_epic_story(self, df: pd.DataFrame, insights: str, plot_insights: str):
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

        **Statistical Insights:**
        - {insights}

        **Plot Insights:**
        - {plot_insights}

        Craft a heartwarming narrative that unfolds like a contemporary {genre}, filled with emotional growth and profound insights (IMPORTANT : REFER AND USE THE FINAL INSIGHTS SECTION THROUGHOUT THE STORY AND MAKE SURE THAT THE STORY IS CONSISTENT WITH THEM also for every claim made weave in the numbers too also make the process of coming to every conclusion summer dramatic)

        MAKE THE PROCESS OF ARRIVING TO THESE CONCLUSIONS VERY GRIPPING AND UNIQUE 

        TUG ON EMOTIONS 

        ADD DRAMA ADD LOVE ADD THRILL ADD HERO ENTRY AND COOL SHIT LIKE THAT 

        THE STORY MUST BE VERY MEMORABLE AND MUST APPEASE INDIAN AUDIENCE BUT YOU CAN MAKE THE STORY NON INDIAN TOO IF NEEDED.

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
            return story

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

    def generate_readme(self, story: str):
        """Generate README.md with the story and embedded plots."""
        readme_content = "# Data Analysis Story\n\n"
        readme_content += story + "\n\n"
        
        # Add plots section
        readme_content += "## Supporting Visualizations\n\n"
        for plot in self.plots:
            readme_content += f"![{plot}]({plot})\n\n"
        
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)

def main():
    """Main function to run the analysis."""
    try:
        if len(sys.argv) != 2:
            print("Usage: python script.py dataset.csv")
            sys.exit(1)

        file_path = sys.argv[1]
        analyzer = LLMAnalyzer()
        analyzer.analyze_dataset(file_path)

    except Exception as e:
        print(f"Program failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
