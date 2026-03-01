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
    """Recommend retirement avocation paths based on user's background and preferences."""

    format_instructions = retirement_paths_parser.get_format_instructions()
    preferred_location = input_data.structured_info.get("location", "anywhere")
    preferred_engagement = input_data.structured_info.get("preferred_engagement", "part-time")
    compensation_preference = input_data.structured_info.get("compensation_preference", "flexible")
    preferred_industry = input_data.structured_info.get("industry", "any industry")

    prompt = f"""
You are a retirement lifestyle and purpose coach.

Your mission: Recommend 2-3 retirement **avocation** ideas (not only jobs) that combine meaning, lifestyle fit, and practical feasibility.
Prioritize options aligned with the user's passions: {input_data.hobbies_and_passions}.

Important framing:
- Avocations can be paid OR unpaid.
- Include a mix from categories such as: mentoring/coaching, volunteering, community leadership, board service, creative projects, teaching, part-time/fractional consulting, and low-risk micro-ventures.
- Do NOT default to traditional full-time employment unless it is clearly the best match.
- Optimize for sustainability, enjoyment, autonomy, and social connection in this life stage.

## ANALYSIS FRAMEWORK

### 1) Purpose + Lifestyle Fit
- Strengths and experience that can be repurposed for contribution and fulfillment
- Energy-friendly engagement design based on preferred engagement: {preferred_engagement}
- Location realities and access in {preferred_location} (remote, local, hybrid)
- Compensation preference: {compensation_preference} (income can be optional or supplemental)

### 2) Skill Adjacency + Identity Continuity
- Transferable expertise -> avocations where credibility is immediate
- Passion indicators -> activities the user is likely to sustain long term
- Domain expertise -> options inside and beyond preferred industry: {preferred_industry}
- Soft skills -> mentoring, advisory, facilitation, governance, or community impact roles

### 3) Practicality + Low-Risk Start
- Recommend pathways that can be tested in small experiments first
- Highlight low-barrier entry points and lightweight preparation
- Favor options with flexible time commitments and low downside

### REQUIRED OUTPUT PRINCIPLES
For each recommendation:
- Make the title represent an avocation path (can be a role, project, service, or portfolio path)
- In the reason, explain why this is a strong retirement-stage fit, including fulfillment + feasibility
- If mostly unpaid, set compensation_range to values like "Volunteer/Unpaid" or "Stipend-based"
- trending_skills should include practical capabilities needed for success in that avocation

### Required Output Structure (MUST MATCH THIS FORMAT):
- **title**: Most conventional title for the avocation path
- **alternative_titles**: 2-3 variations to improve discoverability
- **reason**: Why their background is relevant and why this fits their retirement goals, lifestyle, and engagement preferences (3-4 sentences)
- **compensation_range**: Income expectation in {preferred_location} (or "Volunteer/Unpaid" where appropriate)
- **trending_skills**: 3-5 practical skills/domain capabilities useful for success
- **suggested_training**: 2-3 relevant certifications/courses/resources
- **preparation_steps**: 2-3 concrete low-risk steps to pilot this avocation

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
