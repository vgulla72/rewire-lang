from typing import List
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from models import CareerInput, CareerRecommendationsOutput
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import os

api_key = os.getenv("OPENROUTER_API_KEY")


# Instantiate LLM
#llm = ChatOpenAI(temperature=0, model="gpt-4o")
model_name = "deepseek/deepseek-r1-0528"  # Example model name
llm = ChatOpenAI(
            model=model_name,  # Note: changed from model_name to model
            temperature=0,
            openai_api_base="https://openrouter.ai/api/v1",  # Remove /chat/completions
            openai_api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://rewireme.me",  # Required
                "X-Title": "Resume Analyzer"  # Recommended
            }
        )
# Output parser
career_parser = PydanticOutputParser(pydantic_object=CareerRecommendationsOutput)

def analyze_sectors(sector: str) -> str:
    """Deeply analyze market trends in sector (private, public, academia, nonprofit, freelance) 
    to be able to recommend roles based on individual profiles and interests."""


    prompt = f""" 
Act as a Career and Labor Market Trends Analyst. Generate a detailed, evidence-based report on the key trends, disruptions, and opportunities that will shape job markets in the {sector} over the next 3-5 years:

## ANALYSIS FRAMEWORK
### 1. **Sector Overview**
- Provide a brief overview of the sector, including its current state and significance in the economy.
### 2. **Major Influencing Factors**
- Identify and explain the major factors influencing the sector (e.g., AI, climate change, policy shifts, demographic changes).
### 3. **Emerging Job Roles & Skills in Demand**
- List and describe the new job roles that are emerging in the sector, along with the skills that are increasingly in demand.
### 4. **Declining or At-Risk Jobs**
- Identify jobs that are declining or at risk of being automated or outsourced, and explain why.
### 5. **Geographic Hotspots**
- Highlight regions or countries that are experiencing significant growth in this sector, and explain the reasons for this growth.
### 6. **Work-life Harmony**
- Discuss how work-life balance is evolving in this sector, including trends towards remote work, flexible hours, and employee well-being. What are considerations for someone seeking better work-life harmony in this sector?
- Provide practical advice for job seekers on how to prepare for these changes, including skills to develop, industries to target, and resources to leverage.
### 7. **Sources and References**
- Cite reputable sources, industry reports, and expert projections to support your analysis.
### 8. **Presentation Format**
- Present the findings in a structured, easy-to-read format (bullet points, tables, or summaries).

"""

    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    sector = "freelance"  # Example sector
    result = analyze_sectors(sector)
    print(result)