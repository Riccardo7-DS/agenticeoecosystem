import json
import logging
import re
from pathlib import Path
from agenticeoecosystem import run_gpt
import ollama
import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("PhIdeas MCP Server")
logger = logging.getLogger(__name__)

# Cache for dataframes and processing state
class PhIdeasCache:
    """Cache for PhIdeas data processing"""

    def __init__(self):
        self.dataframes = {}
        self.temperature = 0.5
        self.options = {"temperature": self.temperature}

    def load_dataframe(self, filepath: str) -> pd.DataFrame:
        """Load a dataframe from file and cache it"""
        if filepath in self.dataframes:
            return self.dataframes[filepath]

        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        self.dataframes[filepath] = df
        return df

    def save_dataframe(self, df: pd.DataFrame, filepath: str):
        """Save a dataframe to file and update cache"""
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if filepath.endswith(".csv"):
            df.to_csv(filepath, index=False)
        elif filepath.endswith(".xlsx"):
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        self.dataframes[filepath] = df

    def clean_response(self, res):
        """Extract and return the content from the Ollama response."""
        text = res["response"]
        return text

    def _strip_code_fences(self, s: str) -> str:
        s = str(s).strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-z]*\n?", "", s, flags=re.I)
            s = re.sub(r"\n?```$", "", s)
        return s.strip()

    def _parse_json_like(self, s: str):
        s = self._strip_code_fences(s)
        try:
            return json.loads(s)
        except Exception:
            return []

    def _chunk_indices(self, n: int, size: int):
        i = 0
        while i < n:
            yield range(i, min(i + size, n))
            i += size


# Initialize cache (storing intermediate results when doing multi-step processing)
cache = PhIdeasCache()



@mcp.tool()
def python_tool(code:str) -> str:
    "Executes python code and returns the output"
    local_vars = {}
    try:
        exec(code, {}, local_vars)
        return str(local_vars)
    except Exception as e:
        print(f"Execution error {e}")

@mcp.tool()
def load_excel_first_ten_rows(path:str):
    """ Load and return the first rows of an excel file"""
    import pandas as pd

    df = pd.read_excel(path).head(10)
    return df.to_string()

@mcp.tool()
def clean_expert_column(
    filepath: str, column_name: str = "ESA experts", output_filepath: str = None
) -> str:
    """Clean the expert names column in a PhIdeas dataframe. Extracts only names from text, removes sentences and irrelevant information."""
    try:
        df = cache.load_dataframe(filepath)

        if column_name not in df.columns:
            return f"Error: Column '{column_name}' not found in dataframe. Available columns: {', '.join(df.columns)}"

        def experts_prompt(cell):
            return (
                "Extract only the names of people from the following text. "
                "Remove any sentences, requests, or irrelevant information. "
                "Do not make up names when no names are provided in the text, instead, return an empty string. "
                "The answer should solely be a name, no other text. "
                "When no name is found or value is nan, return an empty string. "
                "Write the name in proper format. "
                "Return a comma-separated list of names only.\n\n"
                f"Text: {cell}"
            )

        def clean_cell(cell):
            if pd.isna(cell) or str(cell).strip() == "":
                return ""
            result = run_gpt(experts_prompt(cell), options=cache.options)
            return cache.clean_response(result)

        df[column_name] = df[column_name].apply(clean_cell)

        if output_filepath is None:
            output_filepath = str(
                Path(filepath).parent / f"cleaned_{Path(filepath).name}"
            )

        cache.save_dataframe(df, output_filepath)

        return f"Successfully cleaned column '{column_name}'. Processed {len(df)} rows. Output saved to: {output_filepath}"

    except Exception as e:
        return f"Error cleaning expert column: {str(e)}"


@mcp.tool()
def summarize_description_column(
    filepath: str,
    source_column: str = "Description",
    target_column: str = "Description Summary",
    output_filepath: str = None,
) -> str:
    """Summarize the description column in a PhIdeas dataframe into concise one-sentence summaries."""
    try:
        df = cache.load_dataframe(filepath)

        if source_column not in df.columns:
            return f"Error: Column '{source_column}' not found in dataframe. Available columns: {', '.join(df.columns)}"

        def summarize_prompt(cell):
            return (
                "Summarize the following text in one clear, concise sentence. "
                "Do not add any extra commentary or information. "
                "Return only the summary sentence.\n\n"
                f"Text: {cell}"
            )

        def process_cell(cell):
            if pd.isna(cell) or str(cell).strip() == "":
                return ""
            result = run_gpt(summarize_prompt(cell), options=cache.options)
            return cache.clean_response(result)

        df[target_column] = df[source_column].apply(process_cell)

        if output_filepath is None:
            output_filepath = str(
                Path(filepath).parent / f"summarized_{Path(filepath).name}"
            )

        cache.save_dataframe(df, output_filepath)

        return f"Successfully summarized column '{source_column}' into '{target_column}'. Processed {len(df)} rows. Output saved to: {output_filepath}"

    except Exception as e:
        return f"Error summarizing description column: {str(e)}"


@mcp.tool()
def categorize_commercial_status(
    filepath: str,
    source_column: str = "Non-Staff - company/institution of origin",
    target_column: str = "Commercial/not",
    output_filepath: str = None,
) -> str:
    """Categorize entities as commercial or non-profit based on their institution/company information."""
    try:
        df = cache.load_dataframe(filepath)

        if source_column not in df.columns:
            return f"Error: Column '{source_column}' not found in dataframe. Available columns: {', '.join(df.columns)}"

        def categorize_prompt(cell):
            return (
                "Classify the entity into two possible categories: commercial, non-profit. "
                "Commercial means the entity has a for-profit purpose, otherwise non-profit. "
                "If you are not able to classify it return Unknown. "
                "Do not add any extra commentary or information. "
                "Return only the category.\n\n"
                f"Text: {cell}"
            )

        def process_cell(cell):
            if pd.isna(cell) or str(cell).strip() == "":
                return "Unknown"
            result = run_gpt(categorize_prompt(cell), options=cache.options)
            return cache.clean_response(result).strip()

        df[target_column] = df[source_column].apply(process_cell)

        if output_filepath is None:
            output_filepath = str(
                Path(filepath).parent / f"categorized_commercial_{Path(filepath).name}"
            )

        cache.save_dataframe(df, output_filepath)

        return f"Successfully categorized column '{source_column}' into '{target_column}'. Processed {len(df)} rows. Output saved to: {output_filepath}"

    except Exception as e:
        return f"Error categorizing commercial status: {str(e)}"


@mcp.tool()
def categorize_cofunding_type(
    filepath: str,
    source_column: str = "Leverage & Co-funding.1",
    target_column: str = "Cofunding Categorization",
    output_filepath: str = None,
) -> str:
    """Categorize cofunding into types: in-kind, financial, in-kind & financial, or none."""
    try:
        df = cache.load_dataframe(filepath)

        if source_column not in df.columns:
            return f"Error: Column '{source_column}' not found in dataframe. Available columns: {', '.join(df.columns)}"

        def categorize_prompt(cell):
            return (
                "Classify the cofunding into four possible contribution categories: in-kind, financial, in-kind & financial, none. "
                "In-kind means the entity gives non-financial resources. "
                "Financial means the entity gives money. "
                "In-kind & financial means the entity gives both non-financial and financial resources. "
                "None means the entity does not provide any resources. "
                "If you are not able to classify it return Unknown. "
                "Do not add any extra commentary or information. "
                "Return only the category.\n\n"
                f"Text: {cell}"
            )

        def process_cell(cell):
            if pd.isna(cell) or str(cell).strip() == "":
                return "Unknown"
            result = run_gpt(categorize_prompt(cell), options=cache.options)
            return cache.clean_response(result).strip()

        df[target_column] = df[source_column].apply(process_cell)

        if output_filepath is None:
            output_filepath = str(
                Path(filepath).parent / f"categorized_cofunding_{Path(filepath).name}"
            )

        cache.save_dataframe(df, output_filepath)

        return f"Successfully categorized cofunding in column '{source_column}' into '{target_column}'. Processed {len(df)} rows. Output saved to: {output_filepath}"

    except Exception as e:
        return f"Error categorizing cofunding type: {str(e)}"


@mcp.tool()
def categorize_funding_binary(
    filepath: str,
    description_column: str = "Description Summary",
    funding_column: str = "Leverage & Co-funding.1",
    target_column: str = "Cofunding binary",
    confidence_column: str = "Confidence_cofunding",
    output_filepath: str = None,
) -> str:
    """Categorize projects as funded/not funded/in-kind with confidence scores."""
    try:
        df = cache.load_dataframe(filepath)

        if description_column not in df.columns:
            return f"Error: Column '{description_column}' not found in dataframe."
        if funding_column not in df.columns:
            return f"Error: Column '{funding_column}' not found in dataframe."

        def categorize_prompt(project, cell):
            return (
                "Classify the project cofunding into three possible categories: funded/ not funded/ in-kind\n"
                "Funded means the initiator presents a figure that represents the financial resources to be allocated to the project in the case the project is approved, not the financial resources allocated in the past.\n"
                "Not funded means the initiator does not contribute to the project with any resources.\n"
                "In-kind means the entity does not give any financial resource but it provides non-financial ones (e.g. computing resources, labor).\n"
                "If you are not able to classify it return Unknown\n"
                "Do not add any extra commentary or information.\n"
                "Return the category and a float for the confidence in classification (from 0 to 1 where 1 means super confident).\n"
                "Return only in the format: <category>||<confidence>.\n\n"
                f"This is the project from the initiator: {project}\n"
                f"This is the funding information: {cell}"
            )

        def process_row(row):
            project = row[description_column]
            funding = row[funding_column]

            if pd.isna(funding) or str(funding).strip() == "":
                return "Unknown", 0.0

            result = run_gpt(categorize_prompt(project, funding), options=cache.options)
            text = cache.clean_response(result).strip()

            try:
                category, confidence = text.split("||")
                category = category.strip()
                confidence = float(confidence.strip())
            except Exception:
                category = "Unknown"
                confidence = 0.0

            return category, confidence

        results = df.apply(process_row, axis=1, result_type="expand")
        df[target_column] = results[0]
        df[confidence_column] = results[1]

        if output_filepath is None:
            output_filepath = str(
                Path(filepath).parent
                / f"categorized_funding_binary_{Path(filepath).name}"
            )

        cache.save_dataframe(df, output_filepath)

        return f"Successfully categorized funding status. Processed {len(df)} rows. Output saved to: {output_filepath}"

    except Exception as e:
        return f"Error categorizing funding binary: {str(e)}"


@mcp.tool()
def categorize_ideas_weighted(
    filepath: str,
    idea_column: str = "Description Summary",
    output_column: str = "Description Categories Weighted",
    max_categories: int = 15,
    max_per_idea: int = 3,
    fallback_column: str = None,
    output_filepath: str = None,
) -> str:
    """Automatically categorize ideas with multiple weighted categories per idea. Proposes categories based on the data."""
    try:
        df = cache.load_dataframe(filepath)

        if idea_column not in df.columns:
            return f"Error: Column '{idea_column}' not found in dataframe. Available columns: {', '.join(df.columns)}"

        ideas = df[idea_column].fillna("").astype(str).tolist()
        if not any(x.strip() for x in ideas):
            return "Error: No valid ideas found in the specified column."

        # Propose categories
        def propose_categories(ideas_list, max_cats):
            prompt = (
                f"You will receive a list of ideas. "
                f"Propose at most {max_cats} concise categories that summarizes the list of ideas. "
                "Keep labels 1 to 4 words. Avoid the word Space. "
                "Avoid explanations, actions, summaries. "
                "Provide only the categories, no other text. "
                'Return a list of discovered categories in this format: ["Category a","Category b","Category c"]\n'
                "Ideas:\n"
                + "\n".join(
                    f"- {str(x)}" for x in ideas_list[:100] if str(x).strip()
                )  # Limit to first 100 for prompt
            )
            res = run_gpt(prompt, options=cache.options)
            text = cache.clean_response(res)
            labels = cache._parse_json_like(text)
            if not isinstance(labels, list):
                labels = ["Other"]
            return labels[:max_cats]

        categories = propose_categories(ideas, max_categories)

        # Assign weighted categories
        def assign_weighted(ideas_list, cats, max_per):
            mapping = {}
            cats_json = json.dumps(cats, ensure_ascii=False)
            for r in cache._chunk_indices(len(ideas_list), 40):
                batch = [ideas_list[i] for i in r]
                batch_text = "\n".join(f"{i + list(r)[0]}. {ideas_list[i]}" for i in r)
                prompt = (
                    f"Assign up to {max_per} categories with weights to each idea.\n"
                    f"Categories: {cats_json}\n"
                    "Return JSON: {row_id: {category: weight, ...}, ...}\n"
                    "Weights should sum to 1.0 per idea.\n\n"
                    f"Ideas:\n{batch_text}"
                )
                res = run_gpt(prompt, options=cache.options)
                text = cache.clean_response(res)
                batch_mapping = cache._parse_json_like(text)
                if isinstance(batch_mapping, dict):
                    mapping.update(batch_mapping)
            return mapping

        assignments = assign_weighted(ideas, categories, max_per_idea)

        # Store as JSON string per row
        df[output_column] = [
            json.dumps(assignments.get(str(i), {"Other": 1.0})) for i in range(len(df))
        ]
        df.attrs["inferred_categories_weighted"] = categories

        # Fallback: assign single category if needed
        if fallback_column:

            def get_top_category(json_str):
                try:
                    cats = json.loads(json_str)
                    return max(cats.items(), key=lambda x: x[1])[0] if cats else "Other"
                except:
                    return "Other"

            df[fallback_column] = df[output_column].apply(get_top_category)

        if output_filepath is None:
            output_filepath = str(
                Path(filepath).parent / f"categorized_weighted_{Path(filepath).name}"
            )

        cache.save_dataframe(df, output_filepath)

        return f"Successfully categorized ideas with weights. Inferred categories: {categories}. Processed {len(df)} rows. Output saved to: {output_filepath}"

    except Exception as e:
        return f"Error categorizing ideas with weights: {str(e)}"


@mcp.tool()
def categorize_ideas_with_imposed_categories(
    filepath: str,
    idea_column: str = "Description Summary",
    output_column: str = "Categories Weighted",
    imposed_categories: str = None,
    max_per_idea: int = 3,
    fallback_column: str = None,
    output_filepath: str = None,
) -> str:
    """Categorize ideas using user-provided categories (comma-separated string) with multiple weighted categories per idea."""
    try:
        df = cache.load_dataframe(filepath)

        if idea_column not in df.columns:
            return f"Error: Column '{idea_column}' not found in dataframe. Available columns: {', '.join(df.columns)}"

        if not imposed_categories:
            return "Error: No categories provided. Please provide a comma-separated string of categories."

        # Parse imposed categories
        categories = [cat.strip() for cat in imposed_categories.split(",")]

        ideas = df[idea_column].fillna("").astype(str).tolist()
        if not any(x.strip() for x in ideas):
            return "Error: No valid ideas found in the specified column."

        # Assign weighted categories
        def assign_weighted(ideas_list, cats, max_per):
            mapping = {}
            cats_json = json.dumps(cats, ensure_ascii=False)
            for r in cache._chunk_indices(len(ideas_list), 40):
                batch_text = "\n".join(f"{i}. {ideas_list[i]}" for i in r)
                prompt = (
                    f"Assign up to {max_per} categories with weights to each idea.\n"
                    f"Categories: {cats_json}\n"
                    "Return JSON: {row_id: {category: weight, ...}, ...}\n"
                    "Weights should sum to 1.0 per idea.\n\n"
                    f"Ideas:\n{batch_text}"
                )
                res = run_gpt(prompt, options=cache.options)
                text = cache.clean_response(res)
                batch_mapping = cache._parse_json_like(text)
                if isinstance(batch_mapping, dict):
                    mapping.update(batch_mapping)
            return mapping

        assignments = assign_weighted(ideas, categories, max_per_idea)

        # Store as JSON string per row
        df[output_column] = [
            json.dumps(assignments.get(str(i), {"Other": 1.0})) for i in range(len(df))
        ]
        df.attrs["imposed_categories_weighted"] = categories

        # Fallback: assign single category if needed
        if fallback_column:

            def get_top_category(json_str):
                try:
                    cats = json.loads(json_str)
                    return max(cats.items(), key=lambda x: x[1])[0] if cats else "Other"
                except:
                    return "Other"

            df[fallback_column] = df[output_column].apply(get_top_category)

        if output_filepath is None:
            output_filepath = str(
                Path(filepath).parent / f"categorized_imposed_{Path(filepath).name}"
            )

        cache.save_dataframe(df, output_filepath)

        return f"Successfully categorized ideas with imposed categories: {categories}. Processed {len(df)} rows. Output saved to: {output_filepath}"

    except Exception as e:
        return f"Error categorizing ideas with imposed categories: {str(e)}"


@mcp.tool()
def get_dataframe_info(filepath: str) -> str:
    """Get information about a PhIdeas dataframe including columns, shape, and sample data."""
    try:
        df = cache.load_dataframe(filepath)

        info = f"Dataframe Information for: {filepath}\n\n"
        info += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        info += "Columns:\n"
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            info += f"  - {col} ({dtype}): {non_null}/{len(df)} non-null values\n"

        info += "\nFirst 3 rows:\n"
        info += df.head(3).to_string()

        return info

    except Exception as e:
        return f"Error getting dataframe info: {str(e)}"


@mcp.tool()
def list_available_columns(filepath: str) -> str:
    """List all available columns in a PhIdeas dataframe."""
    try:
        df = cache.load_dataframe(filepath)
        columns = df.columns.tolist()
        return f"Available columns ({len(columns)}):\n" + "\n".join(
            f"  - {col}" for col in columns
        )
    except Exception as e:
        return f"Error listing columns: {str(e)}"


if __name__ == "__main__":
    mcp.run()
