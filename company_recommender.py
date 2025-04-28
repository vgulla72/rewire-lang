from typing import List
from pydantic import BaseModel
from langchain.tools import tool
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
from career_recommender import CareerInput, CareerRecommendationsOutput

# Define input model
class CompanyInput(BaseModel):
    structured_info: dict
    inferred_insights: dict
    career_change_reason: str
    hobbies_and_passions: str
    career_recommendations: CareerRecommendationsOutput

# Define output model
class CompanyRecommendation(BaseModel):
    company: str
    category: str  # e.g., Non-Profit, For-Profit, Academia, Government
    reason: str

class CompanyRecommendationsOutput(BaseModel):
    recommendations: List[CompanyRecommendation]

# Instantiate LLM
llm = ChatOpenAI(temperature=0, model="gpt-4")

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
    recommend companies or organizations (profit, non-profit, academia, govt) that regularly hire into roles in {formatted_recommendations} and explain why each is a fit.
    Only recommend companies or organizations if they are known to regularly hire for the roles listed above.
    Do NOT suggest companies that do not align with the recommended roles.

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
