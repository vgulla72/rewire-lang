import streamlit as st
from resume_analyzer import ResumeAnalyzer
import tempfile
import os
from career_recommender import recommend_career_paths, CareerInput
from langchain_openai import ChatOpenAI
from langchain.tools import tool

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer")
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
            career_recommendations = recommend_career_paths.invoke({
                "input_data": career_input.model_dump()
            })
            st.subheader("🚀 Career Recommendations")     
            st.write(career_recommendations)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
elif submitted and not uploaded_file:
    st.warning("⚠️ Please upload a PDF file before submitting.")
