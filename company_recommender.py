from typing import List
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from models import CareerInput, CareerRecommendationsOutput, CompanyInput, CompanyRecommendationsOutput, CompanyRecommendation, CareerRecommendation
import os
api_key = os.getenv("OPENROUTER_API_KEY")



# Instantiate LLM
#llm = ChatOpenAI(temperature=0, model="gpt-4o")
model_name = "deepseek/deepseek-r1-0528"  # Example model name
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
parser = PydanticOutputParser(pydantic_object=CompanyRecommendationsOutput)

@tool
def recommend_companies(input_data: CompanyInput) -> CompanyRecommendationsOutput:
    """Recommend companies based on resume context and job openings for roles similar to career recommendations."""

    format_instructions = parser.get_format_instructions()

     # 🧠 Extract the titles and reasons from the career recommendations
    formatted_recommendations = "\n".join([
        f"- {rec.title}: {rec.reason}"
        for rec in input_data.career_recommendations.career_recommendations
    ])
    preferred_location = input_data.structured_info.get("location", "remote")
    preferred_location = "New York, NY"

    prompt = f"""
    You are an expert at finding companies that are currently actively hiring or have in the past hired for recommended roles.

    Based on the following information, recommend companies that:
    1. CURRENTLY HAVE OPEN POSITIONS for roles similar to the {formatted_recommendations}
    2. Are located in {preferred_location} or offer remote work
    3. Match the candidate's profile, experience, and preferences

    IMPORTANT:
    - ONLY recommend companies that you can confirm have active job openings for roles like {formatted_recommendations} in {preferred_location}
    - If you cannot verify current openings, do not recommend the company
    - For each recommendation, specify which recommended role(s) they're hiring for
    - Provide a brief reason for each recommendation, explaining why the company is a good fit

    STRUCTURED INFO:
    {input_data.structured_info}

    INFERRED INSIGHTS:
    {input_data.inferred_insights}

    CAREER CHANGE REASON:
    {input_data.career_change_reason}

    HOBBIES/PASSIONS:
    {input_data.hobbies_and_passions}

    CAREER RECOMMENDATIONS:
    {formatted_recommendations}

    {format_instructions}
    """

    response = llm.invoke(prompt)
    return parser.parse(response.content)
