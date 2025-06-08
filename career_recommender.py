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
def recommend_career_paths(input_data: CareerInput) -> CareerRecommendationsOutput:
    """Recommend career transformations across all sectors (private, public, academia, nonprofit) 
    based on transferable skills and market opportunities."""

    format_instructions = career_parser.get_format_instructions()
    preferred_location = input_data.structured_info.get("location", "anywhere")

    prompt = f""" 
You are a career transition strategist with deep market intelligence across industries and sectors (private, public, non-profit, academia) specializing in non-linear professional transitions. 
Your mission: Identify 3-5 high-potential, career transition opportunities that maximize 
both earning potential and career satisfaction. Analyze this profile through multiple lenses:

## ANALYSIS FRAMEWORK

### 1. **Skill Adjacency Mapping**
- Core transferable skills → Adjacent high-value applications across sectors
- Domain expertise → Cross-industry applications
- Soft skills → Leadership/consulting opportunities
- Technical skills → Emerging tech intersections across industries and sectors

### 2. **Market Intelligence Synthesis**
- Industry convergence points (e.g., FinTech + Healthcare = Digital Therapeutics)
- Regulatory changes creating new roles (e.g., AI governance, data privacy)
- Funding trends indicating growth areas (e.g., climate tech, Web3, biotech)
- Skills arbitrage opportunities (experienced professionals in emerging fields, non-traditional backgrounds in high-demand sectors)

### 3. **Future-Proofing Lens**
- AI-augmented vs. AI-resistant roles
- Remote-first vs. location-dependent opportunities
- Freelance/fractional vs. full-time market dynamics
- Recession-resilient sectors and roles

### 4. **Hidden Market Demand**
- Niche roles companies struggle to fill
- Emerging job categories (created in last 2-3 years)
- Cross-functional hybrid positions
- Consultant-to-employee conversion opportunities
- Cross-sector skills that are undervalued in current market

## CREATIVE EXPLORATION PROMPTS
For each recommendation, consider:
- **Adjacent Industry Jump**: "What if their [current domain] expertise solved problems in [emerging sector]?"
- **Skill Repackaging**: "How could their [specific skill] become a competitive advantage in [growth area]?"
- **Hybrid Role Creation**: "What new role combines their [strength A] + [strength B] + [market need]?"
- **Consulting-to-Employee**: "Which companies would pay premium for their specialized knowledge?"

## CURRENT MARKET INTELLIGENCE (Cross-reference against these trends)
**High-Growth Sectors**: AI/ML, Climate Tech, Digital Health, Cybersecurity, Creator Economy, Web3/Blockchain
**Regulatory Expansion**: AI Ethics, Data Privacy (GDPR/CCPA), ESG Compliance, Crypto Regulation
**Emerging Hybrid Roles**: RevOps, Growth Product Manager, AI Trainer, Sustainability Analyst
**Fractional Executive Boom**: Part-time C-suite, Advisory roles, Specialized consultants


### Required Output Structure (MUST MATCH THIS FORMAT):
- **title**: Most conventional title
- **alternative_titles**: 2-3 variations to improve discoverability (e.g., "Health Tech Advisor", "Digital Health Strategist")
- **reason**: Why their background is relevant (3-4 sentences)
- **compensation_range**: Current salary range (e.g., "$120,000-$160,000")
- **trending_skills**: 3-5 must-have technical/domain skills
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
    return career_parser.parse(response.content)