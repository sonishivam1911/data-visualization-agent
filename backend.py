# backend.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple

# Load the Hugging Face model and tokenizer
MODEL_NAME = "EleutherAI/gpt-neox-20b"  # You can switch this to "bigscience/bloom-7b1" or other models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def parse_user_input(user_input: str) -> str:
    """
    Uses a Hugging Face model to parse the user's natural language input and generate Python code.
    """
    # Define the system instruction
    system_prompt = """
    You are an expert data scientist. Convert the following user request into Python code using pandas and matplotlib/seaborn.
    
    Constraints:
    - Use 'df' as the pandas DataFrame variable.
    - Do not include any import statements.
    - Do not modify or create variables outside the scope of this function.
    - Return only the code necessary to generate the plot.
    
    User Request:
    """
    full_prompt = system_prompt + f"\n\"\"\"\n{user_input}\n\"\"\""

    # Tokenize and generate a response
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs["input_ids"], max_length=300, temperature=0.7)

    # Decode the generated code
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the code portion
    code_start = generated_code.find("def")
    return generated_code[code_start:] if code_start != -1 else generated_code

def execute_generated_code(code: str, df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Executes the generated code safely and returns the matplotlib figure.
    """
    # Define a local namespace for code execution
    local_namespace = {'df': df, 'plt': plt, 'sns': sns}

    try:
        # Execute the code
        exec(code, {'__builtins__': {}}, local_namespace)
        # Retrieve the figure
        fig = plt.gcf()
        return fig
    except Exception as e:
        print(f"Error executing code: {e}")
        return None

def generate_visualization(user_input: str, df: pd.DataFrame) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """
    Orchestrates the parsing and execution of user input to generate a visualization.
    """
    code = parse_user_input(user_input)
    fig = execute_generated_code(code, df)
    return fig, code
    