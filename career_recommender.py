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
    """Recommend private sector career transformations based on transferable skills, 
    emerging industry trends, and adjacent market opportunities."""

    format_instructions = career_parser.get_format_instructions()

    prompt = f"""
You are a radical career transition strategist specializing in identifying high-potential, 
non-obvious private sector opportunities. Analyze this profile through multiple lenses:

1. **Skill Adjacency**: What valuable adjacent roles exist just beyond their current domain?
2. **Industry Convergence**: Where are emerging intersections between their expertise and growing sectors?
3. **Future-Proofing**: What roles leverage both their experience AND future market trends?
4. **Hidden Demand**: What niche positions are companies struggling to fill?

### Creative Framework to Apply:
- "What if" scenarios (e.g., "What if their healthcare tech experience applied to climate tech?")
- Emerging hybrid roles (e.g., "Product Manager + AI Ethicist")
- Skills repackaging (e.g., "Regulatory expertise → Cannabis industry compliance officer")

### Required Output Structure (MUST MATCH THIS FORMAT):
- **title**: Standard LinkedIn job title (e.g., "Healthcare Technology Consultant")
- **reason**: Why their background is relevant (3-4 sentences)
- **compensation_range**: Current salary range (e.g., "$120,000-$160,000")
- **trending_skills**: 3-5 must-have technical/domain skills
- **suggested_training**: 2-3 relevant certifications/courses
- **preparation_steps**: 2-3 actionable steps (e.g., "Get Epic Systems certified")

### Current Industry Trends to Cross-Reference:
- AI-Augmented Roles (e.g., "Prompt Engineering Manager")
- Regulatory Tech Expansion (e.g., "Crypto Compliance Architect")
- Sustainability-Driven Roles (e.g., "Carbon Accounting Specialist")
- Fractional Executive Demand (e.g., "Part-Time CPO for Startups")

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