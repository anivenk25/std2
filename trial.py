import os
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get the token from the environment
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Ensure the token is available
if not AIPROXY_TOKEN:
    print("AIPROXY_TOKEN is not set. Please set it as an environment variable.")
    exit(1)

# Define the proxy API endpoint
PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Define the headers for the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
}

# Function to prompt LLM for initial analysis and code generation
def prompt_llm_for_initial_analysis(data_description):
    prompt = f"""
    Given the following dataset description:

    {data_description}

    Please suggest initial key metrics for analysis (e.g., summary statistics, distributions).
    Also, generate Python code that will calculate those metrics and visualize any basic insights from the dataset.
    """
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data scientist."},
            {"role": "user", "content": prompt}
        ]
    }

    # Make the POST request to the AI Proxy
    response = requests.post(PROXY_URL, headers=headers, data=json.dumps(data))

    # Check for a successful response
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Function to execute the generated Python code
def execute_generated_code(generated_code):
    try:
        # Using exec() to run the code generated by LLM (Note: Be cautious when executing dynamic code!)
        exec(generated_code)
    except Exception as e:
        print(f"Error in executing generated code: {e}")

# Function to analyze and visualize the data
def analyze_and_visualize(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Get a brief description of the dataset (first 3 rows and column types)
    data_description = f"Dataset Columns: {df.columns.tolist()}\n"
    data_description += f"First few rows:\n{df.head(3)}"

    # Prompt the LLM for initial analysis and code generation
    generated_content = prompt_llm_for_initial_analysis(data_description)
    if generated_content:
        print("LLM Generated Insights and Code:\n", generated_content)
        
        # Extract the code from the LLM output
        code_to_execute = generated_content.split("```python\n")[1].split("\n```")[0]  # Extract Python code within ```python``` blocks
        
        # Execute the generated code
        execute_generated_code(code_to_execute)
    else:
        print("Failed to get a response from LLM.")
    
    # Visualizing the data manually in case the LLM doesn't generate code
    visualize_data(df)

    # Recursively ask for deeper insights
    recursive_analysis(df)

# Function to visualize data (manual fallback if LLM doesn't provide visualization code)
def visualize_data(df):
    plt.figure(figsize=(10, 6))
    
    # Visualize distribution of numerical features (histograms for each column)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(num_cols) > 0:
        for col in num_cols:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[col], kde=True, bins=20)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()

    # Visualizing correlation between numerical features (if available)
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

# Function for recursive analysis based on initial insights
def recursive_analysis(df):
    # Prompt LLM to further explore key relationships or metrics
    prompt = f"""
    Based on the initial analysis of the dataset with columns: {df.columns.tolist()},
    which relationships or additional metrics should be explored further? Can you suggest deeper insights or areas of interest?
    """
    generated_content = ask_llm(prompt)
    
    if generated_content:
        print("LLM Suggested Deeper Insights:\n", generated_content)
        
        # Ask the LLM for code to explore these new insights
        prompt_for_code = f"""
        Please generate Python code to explore the following deeper insights or relationships:
        {generated_content}
        """
        generated_code = ask_llm(prompt_for_code)
        
        if generated_code:
            print("LLM Generated Code for Further Insights:\n", generated_code)
            execute_generated_code(generated_code)
        else:
            print("No further code generated.")
    else:
        print("No deeper insights suggested. Proceeding to generate the story.")

# Function to prompt LLM for analysis
def ask_llm(prompt):
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data scientist."},
            {"role": "user", "content": prompt}
        ]
    }

    # Make the POST request to the AI Proxy
    response = requests.post(PROXY_URL, headers=headers, data=json.dumps(data))

    # Check for a successful response
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Function to generate the final story
def prompt_llm_for_story():
    prompt = """
    Based on the analysis of the dataset, generate a detailed story explaining the key insights:
    1. What are the significant findings from the dataset?
    2. Which visualizations best represent these findings?
    3. Can you explain any patterns, correlations, or anomalies in the data?
    4. If applicable, suggest further steps for analysis or model-building.
    """
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data scientist."},
            {"role": "user", "content": prompt}
        ]
    }

    # Make the POST request to the AI Proxy
    response = requests.post(PROXY_URL, headers=headers, data=json.dumps(data))

    # Check for a successful response
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Main function to drive the process
def main(file_path):
    # Initial analysis and visualization
    analyze_and_visualize(file_path)

    # After analysis, prompt for a story
    story = prompt_llm_for_story()
    if story:
        print("\nGenerated Story from LLM:\n", story)

# Example usage
file_path = '/home/anirudh/PycharmProjects/TDS_Project_2/goodreads.csv'  # Replace with the path to your dataset file
main(file_path)

