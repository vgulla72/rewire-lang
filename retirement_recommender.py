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
            temperature=0.8,
            openai_api_base="https://openrouter.ai/api/v1",  # Remove /chat/completions
            openai_api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://rewireme.me",  # Required
                "X-Title": "Resume Analyzer"  # Recommended
            }
        )
# Output parser
retirement_paths_parser = PydanticOutputParser(pydantic_object=CareerRecommendationsOutput)

@tool
def recommend_retirement_paths(input_data: CareerInput) -> CareerRecommendationsOutput:
    """Recommend retirement paths based on user's background and preferences."""

    format_instructions = retirement_paths_parser.get_format_instructions()
    preferred_location = input_data.structured_info.get("location", "anywhere")
    preferred_engagement = input_data.structured_info.get("preferred_engagement", "full-time")
    compensation_preference = input_data.structured_info.get("compensation_preference", "competitive")
    preferred_industry = input_data.structured_info.get("industry", "any industry")

    prompt = f""" 
You are a career coach specializing in transitioning to slowing down and retirement. 
Your mission: Identify 2-3 ideas to explore as possible retirement paths based on the user's background and preferences. 
Prioritize roles that offer the best match with their hobbies {input_data.hobbies_and_passions}. 
For each recommendation, explain why the role is a good fit for the user.
Analyze this profile through multiple lenses:

## ANALYSIS FRAMEWORK

### 1. **Skill Adjacency Mapping**
- Core transferable skills → Adjacent high-value applications across sectors 
- Passion indicators → Roles that align with personal interests {input_data.hobbies_and_passions}
- Domain expertise → Cross-industry applications
- Soft skills → Leadership/consulting opportunities
- Industry preferences -> Do not confine to one industry, explore all industries where their skills are in demand


### 3. **Hidden Market Demand**
- Niche roles companies looking to fill with fractional roles or consulting arrangements
- Volunteer or board opportunities that can leverage their expertise without full retirement
- Passion projects or entrepreneurial ventures that can be started with low risk


### Required Output Structure (MUST MATCH THIS FORMAT):
- **title**: Most conventional title
- **alternative_titles**: 2-3 variations to improve discoverability (e.g., "Health Tech Advisor", "Digital Health Strategist")
- **reason**: Why their background is relevant and explain why the role is a good fit for the user's preferred engagement and compensation expectations. (3-4 sentences)
- **suggested_training**: 2-3 relevant certifications/courses
- **preparation_steps**: 2-3 actionable steps (e.g., "Get Epic Systems certified")


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
    return retirement_paths_parser.parse(response.content)