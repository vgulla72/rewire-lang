import streamlit as st
from resume_analyzer import ResumeAnalyzer
import tempfile
import os
from models import CareerInput, CareerRecommendationsOutput, PeopleSearchInput, CompanyInput, CompanyRecommendationsOutput, PeopleSearchOutput, SectorRecommendationsOutput
from career_recommender import recommend_career_paths
#from career_crew import run_career_crew 
from company_recommender import recommend_companies
from sector_recommender import recommend_sectors
from people_recommender import find_people_transitions
#from people_crew import find_people_transitions
from langchain_openai import ChatOpenAI
from langchain.tools import tool

st.set_page_config(page_title="RewireMe Career Co-pilot", layout="wide")
st.title("📄 RewireMe")
st.markdown("Upload a resume PDF and get structured information and insights.")

# Use a form to control submission
with st.form("resume_form"):
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    reason_for_change = st.text_area("✍️ Reason for career change (optional)")
    hobbies_input = st.text_input("🎯 Hobbies/passions (comma-separated)", placeholder="e.g., photography, mentoring, hiking")

    # Submit button inside the form
    submitted = st.form_submit_button("🔍 Analyze Resume")

if submitted and uploaded_file:
    with st.spinner("Analyzing resume..."):
        # Save the uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        analyzer = ResumeAnalyzer()

        try:
            result = analyzer.analyze(
                file_path=tmp_path
            )

            st.success("✅ Resume successfully analyzed!")
            

            st.subheader("🧾 Structured Resume Info")
            st.json(result["structured_info"])

            st.subheader("💡 Inferred Insights")
            st.json(result["inferred_insights"])

            if reason_for_change.strip():
                st.subheader("🧭 Reason for Career Change")
                st.write(reason_for_change)

            if hobbies_input.strip():
                st.subheader("🏓 Hobbies / Passions")
                st.write(hobbies_input)
            
            # Call the career recommender tool
            career_input = CareerInput(
                structured_info=result["structured_info"],
                inferred_insights=result["inferred_insights"],
                career_change_reason=reason_for_change,
                hobbies_and_passions=hobbies_input
            )
            sector_recommendations = recommend_sectors.invoke({
                "input_data": career_input.model_dump()
            })
            st.subheader("🌐 Sector Recommendations")   
            st.write(sector_recommendations)
            #st.write("-----------")
            
            career_recommendations = recommend_career_paths.invoke({
                "input_data": career_input.model_dump()
            })
            st.subheader("🚀 Career Recommendations")    
            st.write(career_recommendations)
            #st.write("-----------")
            #st.subheader("🔍 Career Recommendations (Detailed View)")
            #career_recommendations_full = recommend_all_careers.invoke({
            #    "input_data": career_input.model_dump()
            #})
            #st.write(career_recommendations_full)

            # Call the company recommender tool
            company_input = CompanyInput(
                structured_info=result["structured_info"],
               inferred_insights=result["inferred_insights"],
                career_change_reason=reason_for_change,
               hobbies_and_passions=hobbies_input,
               career_recommendations=career_recommendations
            )
            company_recommendations = recommend_companies.invoke({
                "input_data": company_input.model_dump()
            })
            st.subheader("🚀 Company Recommendations")     
            st.write(company_recommendations)
            
            # Call the people recommender tool
            st.subheader("🔍 People Search")
            people_input = PeopleSearchInput(
            previous_title=result["structured_info"].get("work_experience", [{}])[0].get("title", ""),
            location=result["structured_info"].get("location", ""),
            recommended_roles=career_recommendations,
            recommended_companies=company_recommendations,
            )
            people_recommendations = find_people_transitions(people_input)
            st.subheader("🚀 People Recommendations")
            st.write(people_recommendations)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

elif submitted and not uploaded_file:
    st.warning("⚠️ Please upload a PDF file before submitting.")
