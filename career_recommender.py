from typing import List
from pydantic import BaseModel
from langchain.tools import tool
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)

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
llm = ChatOpenAI(temperature=0, model="gpt-4")

# Output parser
career_parser = PydanticOutputParser(pydantic_object=CareerRecommendationsOutput)

@tool
def recommend_career_paths(input_data: CareerInput) -> CareerRecommendationsOutput:
    """Recommend career paths based on resume and personal context."""

    format_instructions = career_parser.get_format_instructions()

    prompt = f"""
    Based on the following structured resume info, inferred insights, career change reason, and hobbies/passions,
    suggest 3-4 potential career paths. Suggest only official job titles and do not fabricate roles. Provide reasoning for each suggestion.

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
