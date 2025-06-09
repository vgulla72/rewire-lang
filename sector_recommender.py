from typing import List
from pydantic import BaseModel
from langchain.tools import tool
from langchain.output_parsers import PydanticOutputParser
from models import CareerInput, CareerRecommendationsOutput, SectorRecommendationsOutput, sectorrecommendation
from sector_analyzer import analyze_sectors
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import os

api_key = os.getenv("OPENROUTER_API_KEY")


# Instantiate LLM
#llm = ChatOpenAI(temperature=0, model="gpt-4o")
model_name = "deepseek/deepseek-chat-v3-0324:free"  # Example model name
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
sector_parser = PydanticOutputParser(pydantic_object=SectorRecommendationsOutput)

@tool
def recommend_sectors(input_data: CareerInput) -> SectorRecommendationsOutput:
    """Recommend sectors to focus on based on transferrable skills and career change reason."""

    format_instructions = sector_parser.get_format_instructions()
    career_change_reason = input_data.structured_info.get("career_change_reason", "seeking new opportunities")

    prompt = f""" 
You are a career transition strategist with deep market intelligence across industries and sectors (private, public, non-profit, academia) specializing in non-linear professional transitions. 
Your mission: Identify maximum of 2 sectors that mostly align with {career_change_reason} and maximize 
both earning potential and career satisfaction. 

- Consider the candidate's transferable skills, inferred insights, and career change motivations.
- Suggest 1-2 sectors that are best suited for the candidate's profile, considering their skills, domain, motivations, and market trends.


### Required Output Structure (MUST MATCH THIS FORMAT):
- **sector**: sector name (e.g., private, public, academia, non-profit)
- **reason**: Why this sector is a good fit (3-4 sentences)


### Profile Analysis:
{input_data.structured_info}

### Inferred Superpowers:
{input_data.inferred_insights}

### Change Motivation:
{input_data.career_change_reason}

### Passion Indicators:
{input_data.hobbies_and_passions}

### Format Requirements:
{format_instructions}

"""

    response = llm.invoke(prompt)
    return sector_parser.parse(response.content)