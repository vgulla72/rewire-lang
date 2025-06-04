from typing import List
from pydantic import BaseModel
from langchain.tools import tool
from langchain.output_parsers import PydanticOutputParser
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from models import CareerInput, CareerRecommendationsOutput, CompanyInput, CompanyRecommendationsOutput, CompanyRecommendation, CareerRecommendation
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
parser = PydanticOutputParser(pydantic_object=CompanyRecommendationsOutput)

@tool
def recommend_companies(input_data: CompanyInput) -> CompanyRecommendationsOutput:
    """Recommend companies based on resume and context and career recommendations."""

    format_instructions = parser.get_format_instructions()

     # 🧠 Extract the titles and reasons from the career recommendations
    formatted_recommendations = "\n".join([
        f"- {rec.title}: {rec.reason}"
        for rec in input_data.career_recommendations.career_recommendations
    ])

    prompt = f"""
    You are a company recommendation expert.

    Based on the structured resume info, inferred insights, career change reason, and hobbies/passions,
    recommend companies that hire into roles in {formatted_recommendations} and explain why each is a fit.
    Only recommend companies or organizations if they align with the work environment and cultural factors important to the user.
    Recommend off the beaten path companies that are not commonly found on LinkedIn or job boards.
    Include the work environment and cultural factors considered for each recommendation.

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
