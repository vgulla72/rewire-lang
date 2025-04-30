from typing import List
from pydantic import BaseModel
from langchain.tools import tool
from langchain.output_parsers import PydanticOutputParser
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import os

api_key = os.getenv("OPENROUTER_API_KEY")

# Pydantic model to structure input
class CareerInput(BaseModel):
    structured_info: dict
    inferred_insights: dict
    career_change_reason: str
    hobbies_and_passions: str

# Define output model
class CareerRecommendation(BaseModel):
    title: str
    reason: str

class CareerRecommendationsOutput(BaseModel):
    career_recommendations: List[CareerRecommendation]

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
    """Recommend career paths based on resume and personal context."""

    format_instructions = career_parser.get_format_instructions()

    prompt = f"""
    You are a career advisor AI recommending new career paths to a professional considering a transition. 
    Use the structured resume info, inferred insights, career change reason, and personal interests below.

    Recommend 3–4 potential career paths that:
    - Are a strong fit for their background and goals
    - Suggest official job titles** that are commonly found on LinkedIn and job boards  (e.g., if the recommended role is "Technology Consultant - Healthcare", suggest normalized title like "Technology Consultant")
    - Are written to **maximize searchability and discoverability** when looking for people in those roles on LinkedIn
    - Include a brief explanation of why each career path is a good fit

    Do NOT simply restate the same job title unless it's already a highly normalized or widely used form.

    Structured Info:
    {input_data.structured_info}

    Inferred Insights:
    {input_data.inferred_insights}

    Career Change Reason:
    {input_data.career_change_reason}

    Hobbies/Passions:
    {input_data.hobbies_and_passions}

    {format_instructions}
    """

    response = llm.invoke(prompt)
    return career_parser.parse(response.content)
