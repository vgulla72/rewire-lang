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
model_name = "deepseek/deepseek-chat-v3-0324:free"  # Example model name
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
def recommend_pivot_paths(input_data: CareerInput, sector: str, sector_analysis: List[str]) -> CareerRecommendationsOutput:
    """Recommend non-linear career pivots into the {sector} sector based on transferable strengths, passion indicators, {sector_analysis}, and market demand."""

    format_instructions = career_parser.get_format_instructions()
    preferred_location = input_data.structured_info.get("location", "anywhere")
    preferred_engagement = input_data.structured_info.get("preferred_engagement", "full-time")
    compensation_preference = input_data.structured_info.get("compensation_preference", "competitive")

    prompt = f""" 
Your mission: Recommend 2–3 **non-linear, high-potential career pivot opportunities** within the {sector} sector for this individual. These should not assume a direct step forward in their current career path, but instead reflect creative, adjacent, or cross-sector leaps that make use of their **transferable strengths**, **motivation for change**, and **passion indicators** ({input_data.hobbies_and_passions}).

Focus on:
- Roles they may not have considered but are an unexpected fit
- How their **soft skills, leadership traits, or cross-domain experience** could shine
- Opportunities for **identity-shifting roles** (e.g., teacher → UX researcher, ops lead → AI ethics consultant)
- Hybrid jobs and interdisciplinary roles gaining traction in {sector}
-Only recommend roles that are actively being hired for today. Validate each role title and at least one alternative title against real job listings in {preferred_location} using live job search.
Analyze this profile through multiple lenses:

## ANALYSIS FRAMEWORK

### 1. **Career Reframing Through Transferable Strengths**
- Core strengths → Surprising yet strategic pivot roles
- Engagement preferences → Non-traditional or hybrid roles supporting {preferred_engagement}
- Consider Compensation expectations {compensation_preference} → Evaluate alignment even if pivot includes a short-term tradeoff
- Soft skills and personal interests → Leverage to uncover creative fit
- IC vs. Managerial flexibility → Explore both tactical and strategic paths
- Domain overlap → Look for unusual entry points from previous roles
- Passion indicators → Anchor in interests like {input_data.hobbies_and_passions}

### 2. **Market Intelligence Synthesis**
- Understand and creatively apply the {sector_analysis}

### 3. **Hidden Market Demand**
- Niche or undervalued cross-functional roles
- Interdisciplinary opportunities with low traditional barriers to entry
- Roles where past experience provides unusual credibility
- Career pivots common among mid-career professionals
- Consultant-to-employee transitions or fractional roles that open full-time pathways


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