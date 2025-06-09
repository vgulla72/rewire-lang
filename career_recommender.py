from typing import List
from pydantic import BaseModel
from langchain.tools import tool
from langchain.output_parsers import PydanticOutputParser
from models import CareerInput, CareerRecommendationsOutput
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
career_parser = PydanticOutputParser(pydantic_object=CareerRecommendationsOutput)

@tool
def recommend_career_paths(input_data: CareerInput, sector: str, sector_analysis: List[str]) -> CareerRecommendationsOutput:
    """Recommend career transformations in the {sector} sector
    based on transferable skills, {sector_analysis} and market opportunities."""

    format_instructions = career_parser.get_format_instructions()
    preferred_location = input_data.structured_info.get("location", "anywhere")

    prompt = f""" 
You are a career transition strategist with deep market intelligence across industries and sectors (private, public, non-profit, academia) specializing in non-linear professional transitions. 
Your mission: Identify 2-3 high-potential, career transition opportunities within {sector} that maximize 
both their motivation to change and career and compensation aspirations. Analyze this profile through multiple lenses:

## ANALYSIS FRAMEWORK

### 1. **Skill Adjacency Mapping**
- Core transferable skills → Adjacent high-value applications across sectors
- IC vs. Managerial roles → Leadership/management opportunities
- Typical compensation ranges → Salary expectations
- Career aspirations → Growth trajectories in {sector} 
- Passion indicators → Roles that align with personal interests
- Domain expertise → Cross-industry applications
- Soft skills → Leadership/consulting opportunities
- Technical skills → Emerging tech intersections across industries and sectors

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
- **reason**: Why their background is relevant (3-4 sentences)
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