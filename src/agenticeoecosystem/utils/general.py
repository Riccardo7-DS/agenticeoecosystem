import json
import ast
import re
import pandas as pd
import logging 
import json
import ollama
import re
import ast
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

def run_code(code):
    local_vars = {}
    try:
        exec(code, {}, local_vars)
    except Exception as e:
        print(f"Execution error {e}")
        print("Generated code was:", code)
    return local_vars

def clean_code(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("```")[1]
    return code.strip()

python_tool = RunnableLambda(run_code)


async def print_and_execute(code, debug=True):
    if debug:
        print("Runnable lambda OUTPUT:")
        # print(code)

    code_text = extract_python_code(code)
    result = python_tool.invoke(code_text)

    if debug:
        print("\nEXTRACTED CODE:")
        print(result)

    return result


def extract_python_code(text: str) -> str:
    # Try to extract ```python blocks
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code block, try to detect if entire text is valid Python
    try:
        ast.parse(text)
        return text.strip()
    except SyntaxError:
        pass

    raise ValueError("No executable Python code found in LLM output.")


def extract_json_from_text(text: str):
    """
    Attempt to extract a JSON object or array from arbitrary LLM output.
    Returns the JSON as Python object, or None if not found.
    """
    # Try to extract the first {...} or [...] block
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return None

    json_text = match.group(0)

    # Try strict json.loads first
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass

    # Fallback: remove newlines and backticks
    json_text_clean = json_text.replace("`", "").replace("\n", " ")
    try:
        return json.loads(json_text_clean)
    except json.JSONDecodeError:
        return None

# def run_gpt(question, model):
#     """Run a local GPT chat completion with the given question using Ollama."""
#     result = ollama.generate(model=model, prompt=question)
#     if not result or 'response' not in result:
#         raise ValueError("GPT response is None or invalid")
#     return result


# Helper function to run GPT
def run_gpt(question, options={"temperature": 1}, model="gpt-oss:20b"):
    """Run a local GPT chat completion with the given question using Ollama."""
    result = ollama.generate(model=model, prompt=question, options=options)
    if not result or "response" not in result:
        raise ValueError("GPT response is None or invalid")
    return result

class OllamaQuery():
    def __init__(self, model):
        self.model = model

    def clean_response(self, res):
        """Extract and return the content from the Ollama response."""
        return res['response']

    def save_response_to_csv(self, response, file_name):
        """Save the response to a CSV file."""
        df = pd.DataFrame({"Response": [response]})
        df.to_csv(file_name, index=False)

    def clean_cell(self, cell, prompt_template):
        from agenticeoecosystem import run_gpt
        logger.info("\n-----------------------")
        logger.info(f"\nOriginal cell value: {cell}")
        prompt = prompt_template(cell)
        result = run_gpt(prompt)
        cleaned = self.clean_response(result)
        logger.info(f"Cleaned cell value: {cleaned}")
        return cleaned

    def clean_column(self, df, column, prompt_template):
        df[column] = df[column].apply(lambda cell: self.clean_cell(cell, prompt_template))
        return df

    def add_column(self, df, source_col, new_col, prompt_template):
        df[new_col] = df[source_col].apply(lambda cell: self.clean_cell(cell, prompt_template))
        return df

    def summarize_prompt(self, cell):
        return (
            "Summarize the following text in one clear, concise sentence. "
            "Do not add any extra commentary or information. "
            "Return only the summary sentence.\n\n"
            f"Text: {cell}"
        )

    ### Idea categorization
    def _strip_code_fences(self, s: str) -> str:
        s = str(s).strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        return s.strip()


    def _parse_json_like(self, s: str):
        s = self._strip_code_fences(s)
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
                if not m:
                    raise ValueError("Could not parse model output as JSON or Python literal.")
                block = m.group(1)
                try:
                    return json.loads(block)
                except Exception:
                    return ast.literal_eval(block)


    def _chunk_indices(self, n: int, size: int):
        i = 0
        while i < n:
            yield range(i, min(i + size, n))
            i += size

