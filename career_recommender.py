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

    prompt = f""" 
You are a career transition strategist with deep market intelligence and expertise across private, public, academia, nonprofit sectors specializing in non-linear professional transitions. 
Your mission: Identify 5-7 high-potential, non-obvious opportunities that maximize career satisfaction. Analyze this profile through multiple lenses:

For each sector, analyze through these lenses:
- **Private Sector**: Corporate roles, startups, consulting, fractional executive positions
- **Public Sector**: Government, policy, civil service, regulatory roles
- **Academia**: Research, teaching, administration, edtech
- **Nonprofit**: Social impact, advocacy, fundraising, program management

### 2. **Cross-Sector Skill Translation**
- Identify how core skills transfer differently across sectors
- Highlight sector-specific value propositions for the candidate's background
- Note any certifications or training needed for sector transitions

### 3. **Compensation & Lifestyle Factors**
- Compare earning potential across sectors
- Consider stability vs. growth tradeoffs
- Evaluate work-life balance considerations

### 4. **Impact Potential**
- Map opportunities to candidate's change motivation and passions
- Identify roles with highest alignment to personal values

## SECTOR-SPECIFIC TRENDS (2024)

**Private Sector Trends**:
- AI integration across business functions
- Growth of hybrid tech/business roles
- Specialized consulting in regulatory changes

**Public Sector Trends**:
- Digital transformation in government
- Climate policy implementation roles
- Cybersecurity in public infrastructure

**Academic Trends**:
- Interdisciplinary research growth
- Edtech and digital learning innovation
- Science communication roles

**Nonprofit Trends**:
- Data-driven impact measurement
- Corporate partnership roles
- Policy advocacy in tech regulation

## CREATIVE EXPLORATION PROMPTS
For each recommendation, consider:
- **Sector Transition**: "How could their skills solve problems differently in [sector]?"
- **Impact Multiplier**: "Which roles would amplify both their skills and desired impact?"
- **Compensation Strategy**: "What roles offer the best reward for their unique skill mix?"
- **Adjacent Industry Jump**: "What if their [current domain] expertise solved problems in [emerging sector]?"
- **Skill Repackaging**: "How could their [specific skill] become a competitive advantage in [growth area]?"
- **Hybrid Role Creation**: "What new role combines their [strength A] + [strength B] + [market need]?"
- **Consulting-to-Employee**: "Which companies would pay premium for their specialized knowledge?"
- **Nonprofit Innovation**: "How could their [skill] drive change in [social issue]?"
- **Academic-to-Industry**: "What industry roles value their research expertise?"
- **Public Sector Innovation**: "How could their [skill] transform public services in [area]?"
- **Startup Opportunities**: "What gaps in [industry] could they fill with their unique background?"
- **Fractional Executive Roles**: "What companies need their expertise on a part-time basis?"
- **Remote Work Trends**: "How can their skills adapt to the growing remote work landscape?"
- **Gig Economy Roles**: "What freelance opportunities align with their expertise?"

### Required Output Structure (MUST MATCH THIS FORMAT):
- **category**: Sector (Private/Public/Academia/Nonprofit)
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