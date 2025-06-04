from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from models import PeopleSearchInput, PeopleSearchOutput
from tools import serper_search  # Assuming it's in a tools.py or same file
import os

api_key = os.getenv("OPENROUTER_API_KEY")

# Instantiate LLM
#llm = ChatOpenAI(temperature=0, model="gpt-4o")
model_name = "meta-llama/llama-4-maverick:free"  # Example model name
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

def find_people_transitions(input_data: PeopleSearchInput) -> PeopleSearchOutput:
    previous_title = input_data.previous_title
    location = input_data.location
    role_titles = [r.title for r in input_data.recommended_roles.career_recommendations]

    queries = [
        f'site:linkedin.com/in "{previous_title}" AND "{role}" AND "{location}"'
        for role in role_titles
    ]

    agent = Agent(
        role="Career Transition Researcher",
        goal="Find real-life people who have made a career transition relevant to the input",
        backstory="You are a talent researcher specializing in identifying professional transitions based on online profiles and web results.",
        tools=[serper_search],
        llm=llm,
        verbose=True
    )

    task = Task(
        description=f"""Use web search (via the provided tool) to identify people who have gone from '{previous_title}' to any of the roles: {', '.join(role_titles)} in {location}. Extract 3–5 matching examples with name, titles, company, LinkedIn URL, and summary.""",
        agent=agent,
        expected_output="JSON array of people: name, previous_title, current_title, current_company, linkedin_profile, summary",
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    
    parser = PydanticOutputParser(pydantic_object=PeopleSearchOutput)
    try:
        return parser.parse(result)
    except Exception:
        # Basic fallback if the output is invalid JSON
        return PeopleSearchOutput(matches=[])
