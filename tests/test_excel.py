from agenticeoecosystem import run_eve_agent, run_code_pipeline
import logging

import pandas as pd 

logger = logging.getLogger(__name__)

model ="llama3.1:8b"
# model="mistral"
# model = "glm-4.7-flash:q8_0" 

filepath = "/home/egm/Desktop/agenticecosystems/data/Final_assignment.xlsx"
# data = pd.read_excel(filepath, sheet_name="Final_assignment", header=1)

prompt = f"""
Open in pandas the file in the filepath {filepath} and the sheet name "Final_assignment". The headers are in the second row.
Return the first rows of the pandas dataframe
"""

# You MUST use the provided tools to access local files.
# You cannot access files without calling a tool.
# Create a subset df with the columns that approximately match the title name "Idea ID" and "Confidence in reviewing". Rename the columns to these names.
# Give me a print of subsets in pandas df as a result. If you do not find the columns, do not provide anything"
# DO not give me a fake table. 

# prompt = f"""
# Open file in the filepath {filepath} and the sheet "Final_assignment. The headers are in the second row".
# Find the columns that approximately match the title name "Idea ID" and "Confidence in reviewing"
# Return a dataframe with these columns only.
# """



# asyncio.run(run_eve_agent(prompt, model, temperature=0))


import asyncio
asyncio.run(run_eve_agent(model_name=model, user_prompt=prompt, temperature=0))
