from langchain.tools import tool
from pydantic import BaseModel
from typing import Optional
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

# Pydantic model to structure input
class CareerInput(BaseModel):
    structured_info: dict
    inferred_insights: dict
    career_change_reason: Optional[str] = None
    hobbies_and_passions: Optional[str] = None

@tool
def recommend_career_paths(input_data: CareerInput) -> str:
    """Recommend career paths based on resume and personal context."""
    
    prompt = f"""
    Based on the following structured resume info, inferred insights, career change reason, and hobbies/passions,
    suggest 3-4 potential career paths. Provide reasoning for each suggestion.

    Structured Info:
    {input_data.structured_info}

    Inferred Insights:
    {input_data.inferred_insights}

    Career Change Reason:
    {input_data.career_change_reason}

    Hobbies/Passions:
    {input_data.hobbies_and_passions}

    Format your response as:
    1. Role: ...
       Why: ...
    2. Role: ...
       Why: ...
    """

    response = llm.invoke(prompt)
    return response.content
