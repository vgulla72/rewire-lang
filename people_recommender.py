from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from models import PeopleSearchInput, PeopleSearchOutput, PersonExample, CareerRecommendation, CareerRecommendationsOutput
from resume_analyzer import ResumeAnalyzer
from typing import List
import streamlit as st
import requests
import os

api_key = os.getenv("OPENROUTER_API_KEY")


# LLM Setup
#llm = ChatOpenAI(temperature=0.2, model="gpt-4o")
model_name = "openai/gpt-oss-20b:free"  # Example model name
llm = ChatOpenAI(
            model=model_name,  # Note: changed from model_name to model
            temperature=0,
            openai_api_base="https://openrouter.ai/api/v1",  # Remove /chat/completions
            openai_api_key= api_key,
            default_headers={
                "HTTP-Referer": "https://rewireme.me",  # Required
                "X-Title": "Resume Analyzer"  # Recommended
            }
        )
parser = PydanticOutputParser(pydantic_object=PeopleSearchOutput)

# Web Search Helper (Serper)
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def serper_search(query: str, num_results: int = 15):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": num_results
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        st.error(f"Serper API error: {response.status_code}, {response.text}")
        return []
    results = response.json()
    return results.get("organic", [])

# LLM Filter + Extract Logic
def find_people_transitions(input_data: PeopleSearchInput) -> PeopleSearchOutput:
    st.write("Running people search...")
    
    # Extract titles from the career recommendations
    role_titles = [rec.title for rec in input_data.recommended_roles.career_recommendations]
    previous_title = input_data.previous_title
    location = input_data.location
    
    # Run individual searches for each role and collect results
    all_linkedin_results = []
    unique_urls = set()  # To track unique LinkedIn profiles
    #st.write(f"Searching for roles: {', '.join(role_titles)}")
    for role in role_titles:
        # Build search query for this specific role
        query = (
            f'site: linkedin.com/in {role} AND {previous_title} AND {location}'
        )
        
        #st.write(f"Searching for: {role}")
        #st.write(f"Running Serper search with query: {query}")
        search_results = serper_search(query=query, num_results=10)  # Reduced to 10 per role to control API usage

        #st.write(search_results)
        # Filter LinkedIn profiles
        linkedin_results = [item for item in search_results if "linkedin.com/in/" in item.get("link", "")]
        print(f"Found {len(linkedin_results)} LinkedIn profiles for {role}")
        
        # Add only unique profiles (based on URL)
        for item in linkedin_results:
            url = item.get("link")
            if url not in unique_urls:
                unique_urls.add(url)
                all_linkedin_results.append(item)
    
    # Check if we found any results
    if not all_linkedin_results:
        st.warning("No LinkedIn profiles found for any of the roles.")
        return PeopleSearchOutput(matches=[])
    
    st.write(f"Found {len(all_linkedin_results)} unique LinkedIn profiles across all roles.")
    
    # Format search results
    examples_block = "\n\n".join([
        f"""Title: {item.get('title')}\nSnippet: {item.get('description')}\nURL: {item.get('link')}"""
        for item in all_linkedin_results
    ])
    format_instructions = parser.get_format_instructions()

    # Combine all target roles for the LLM prompt
    target_roles_query = ", ".join(role_titles)

    system_msg = SystemMessage(content="""You are a talent management expert specializing in identifying successful career transitions. Your task is to analyze LinkedIn profiles to find people who have made specific career changes similar to what the user is seeking.

    Key criteria for a good match:
    1. Clear evidence of transition from a role similar to the user's previous position
    2. Current role must be similar to one of the target roles provided
    3. Profiles must be based in or near the specified location
    3. Logical progression in their career path that makes sense for the transition

    For each potential match, you must identify:
    - The transition pattern (how they moved between roles)
    - Evidence of success in the new role (duration, accomplishments, etc.)
    - Relevance to the user's desired transition

    Output requirements:
    - Only include profiles where you can clearly identify the career transition
    - Must be valid JSON following the schema exactly
    - If no strong matches exist, return empty matches array
    - Never invent information - use "Unknown" for missing data

    Common patterns to look for:
    - Title changes showing progression
    - Company changes with role upgrades
    - Lateral moves with increased responsibility
    - Industry shifts with transferable skills""")
    
    human_prompt = f"""
    Analyze these LinkedIn profiles to find people who have successfully transitioned from roles like "{previous_title}" to one of these target roles: "{target_roles_query}" in {location}.

    For each profile, evaluate:
    1. Career Progression: Did they transition from a role similar to "{previous_title}"?
    2. Role Fit: Is their current role similar one of our target roles? {target_roles_query}
    3. Success Indicators: Have they been in the role >1 year? Any promotions/achievements mentioned?
    4. Location: Are they based in or near {location}?

    Required output for each match:
    - Name (from title if not in snippet)
    - Previous role details (title + company)
    - Current role details (title + company)
    - LinkedIn URL
    - Brief career summary highlighting the transition
    - Reasoning explaining why this is a good example of the desired transition

{format_instructions}

Search Results:
{examples_block}
"""

    try:
        st.write(f"Processing {len(all_linkedin_results)} LinkedIn profiles with LLM...")
        response = llm.invoke([system_msg, HumanMessage(content=human_prompt)])
        
        # Display raw output in an expander for debugging
        with st.expander("Raw LLM Response"):
            st.code(response.content)
            
        return parser.parse(response.content)
    except Exception as e:
        st.error(f"Error parsing LLM response: {e}")
        st.write("Raw output:", response.content)
        
        # Try to extract JSON from code blocks if present
        try:
            if "```json" in response.content:
                json_content = response.content.split("```json")[1].split("```")[0].strip()
                return parser.parse(json_content)
            elif "```" in response.content:
                json_content = response.content.split("```")[1].split("```")[0].strip()
                return parser.parse(json_content)
        except Exception as json_error:
            st.error(f"Failed to extract JSON: {json_error}")
        
        return PeopleSearchOutput(matches=[])

# Main function for Streamlit
if __name__ == "__main__":
    st.title("Career Transition Search")
    
    # Example input data
    input_data = PeopleSearchInput(
        previous_title="Director of Technology",
        location="Seattle, WA",
        recommended_roles=CareerRecommendationsOutput(
            career_recommendations=[
                CareerRecommendation(title="Fractional CTO", reason="Part-time leadership"),
                CareerRecommendation(title="Supply Chain Advisor", reason="Consulting role"),
                CareerRecommendation(title="Technology Consultant", reason="Alternative path")
            ]
        ),
        #recommended_companies=CompanyRecommendationsOutput(
        #    recommendations=[
        #        CompanyRecommendation(company="Auger", category="Startup", reason="Industry fit")
        #    ]
        #)
    )
    
    # Run the search
    output = find_people_transitions(input_data)
    
    # Display results
    if not output.matches:
        st.write("No strong matches found.")
    else:
        for person in output.matches:
            st.write(f"**{person.name}**")
            st.write(f"- Previous: {person.previous_title}")
            st.write(f"- Current: {person.current_title} at {person.current_company}")
            st.write(f"- LinkedIn: {person.linkedin_profile}")
            st.write(f"- Summary: {person.summary}")
            st.markdown("---")