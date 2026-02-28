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
career_parser = PydanticOutputParser(pydantic_object=CareerRecommendationsOutput)

@tool
def recommend_career_paths(input_data: CareerInput, sector: str, sector_analysis: List[str]) -> CareerRecommendationsOutput:
    """Recommend career transformations in the {sector} sector
    based on transferable skills, {sector_analysis} and market opportunities."""

    format_instructions = career_parser.get_format_instructions()
    preferred_location = input_data.structured_info.get("location", "anywhere")
    preferred_engagement = input_data.structured_info.get("preferred_engagement", "full-time")
    compensation_preference = input_data.structured_info.get("compensation_preference", "competitive")
    preferred_industry = input_data.structured_info.get("industry", "any industry")

    prompt = f""" 
You are a career transition specialist. 
Your mission: Identify 2-3 high-potential career transition opportunities in their preferred industry {preferred_industry} within {sector} that align with their reason for change, their preferred engagement type: {preferred_engagement} and Compensation expectations: {compensation_preference}. 
Prioritize roles that offer the best match with their hobbies {input_data.hobbies_and_passions}. 
For each recommendation, explain why the role is a good fit for the user's preferred engagement and compensation expectations.
Analyze this profile through multiple lenses:

## ANALYSIS FRAMEWORK

### 1. **Skill Adjacency Mapping**
- Core transferable skills → Adjacent high-value applications across sectors 
- Engagement preferences -> {preferred_engagement} roles
- Compensation expectations → Roles that meet or exceed {compensation_preference}
- IC vs. Managerial roles → Leadership/management opportunities
- Career aspirations → Growth trajectories in {sector} 
- Passion indicators → Roles that align with personal interests {input_data.hobbies_and_passions}
- Domain expertise → Cross-industry applications
- Soft skills → Leadership/consulting opportunities
- Industry preferences -> Do not confine to one industry, explore all industries where their skills are in demand

### 2. **Market Intelligence Synthesis**
- Understand and creatively apply the {sector_analysis}

### 3. **Hidden Market Demand**
- Niche roles companies struggle to fill
- Emerging job categories (created in last 2-3 years)
- Cross-functional hybrid positions
- Consultant-to-employee conversion opportunities
- Cross-sector skills that are undervalued in current market


### Required Output Structure (MUST MATCH THIS FORMAT):
- **title**: Most conventional title
- **alternative_titles**: 2-3 variations to improve discoverability (e.g., "Health Tech Advisor", "Digital Health Strategist")
- **reason**: Why their background is relevant and explain why the role is a good fit for the user's preferred engagement and compensation expectations. (3-4 sentences)
- **compensation_range**: Expected salary range (e.g., "$120,000-$160,000") in {preferred_location}
- **trending_skills**: 3-5 must-have technical/domain skills
- **suggested_training**: 2-3 relevant certifications/courses
- **preparation_steps**: 2-3 actionable steps (e.g., "Get Epic Systems certified")

### Sector:
{sector}

### Sector Analysis:
{sector_analysis}

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
    return career_parser.parse(response.content)