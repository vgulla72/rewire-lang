from typing import List
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from models import question_answer, question_input, daily_tasks
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import os

api_key = os.getenv("OPENROUTER_API_KEY")


# Instantiate LLM
#llm = ChatOpenAI(temperature=0, model="gpt-4o")
model_name = "meta-llama/llama-3.3-70b-instruct:free"  # Example model name
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
question_parser = PydanticOutputParser(pydantic_object=question_answer)
daily_tasks_parser = PydanticOutputParser(pydantic_object=daily_tasks)

@tool
def generate_questions(input_data: question_input) -> question_answer:
    """Generate questions based on the input topic."""

    format_instructions = question_parser.get_format_instructions()
    topic = input_data.topic
    age = input_data.age
    gender = input_data.gender
    # update prompt to request multiple choice options as well
    prompt = f"""
You are a quiz master tasked with generating a multiple-choice question based on the given topic appropriate for person's age and gender.
Your mission: Generate a relevant question, a list of options (at least four) and clearly indicate the correct answer.

### Required Output Structure (MUST MATCH THIS FORMAT):
- **question**: The generated question
- **options**: A list of possible choices (include the correct answer among them)
- **answer**: The correct answer text (must exactly match one of the options)


### Topic:  {topic}
### Age: {age}  
### Gender: {gender}

### Format Requirements:
{format_instructions}

"""

    response = llm.invoke(prompt)
    return question_parser.parse(response.content)

@tool
def daily_tasks(input_data: question_input) -> daily_tasks:
    """Generate daily tasks based on the input data."""

    format_instructions = daily_tasks_parser.get_format_instructions()
    topic = input_data.topic
    age = input_data.age
    gender = input_data.gender
    # update prompt to request multiple choice options as well
    prompt = f"""
You are a therapist specialised in treating dementia patients and generating daily tasks based person's age, gender, location, religion to improve their cognition and memory.
Your mission: Generate a list of daily tasks that are suitable for a dementia patient.

### Required Output Structure (MUST MATCH THIS FORMAT):
- **tasks**: A list of daily tasks that are suitable for a dementia patient


### Age: {age}  
### Gender: {gender}
### Location: Rajahmundry, Andhra Pradesh, India
### Religion: Hinduism

### Format Requirements:
{format_instructions}

"""

    response = llm.invoke(prompt)
    return daily_tasks_parser.parse(response.content)