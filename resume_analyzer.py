import os
import pdfplumber
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from models import StructuredResumeInfo, InferredProfileInsights

api_key = os.getenv("OPENROUTER_API_KEY")

class ResumeAnalyzer:
    def __init__(self, model_name="deepseek/deepseek-chat-v3-0324:free", temperature=0):
        # Configure OpenRouter
        self.llm = ChatOpenAI(
            model=model_name,  # Note: changed from model_name to model
            temperature=temperature,
            openai_api_base="https://openrouter.ai/api/v1",  # Remove /chat/completions
            openai_api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://rewireme.me",  # Required
                "X-Title": "Resume Analyzer"  # Recommended
            }
        )
        self.structured_parser = PydanticOutputParser(pydantic_object=StructuredResumeInfo)
        self.insight_parser = PydanticOutputParser(pydantic_object=InferredProfileInsights)

    def parse_pdf(self, file_path: str) -> str:
        """Extracts text content from a PDF resume."""
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text

    def extract_structured_info(self, resume_text: str) -> StructuredResumeInfo:
        """Uses LLM to extract structured resume info."""
        prompt = f"""
        Extract structured information from the following resume and calculate total years of experience based on the work experience section.
        Your task is to extract **structured information** from the following resume. If any field is missing, leave it blank or use null. Calculate total years of experience based on the work experience durations.

        Extract **only** the following fields:
        - full_name (string)
        - email (string)
        - phone_number (string or null)
        - location (string)
        - work_experience (list of objects: {{title, company, duration}})
        - education (list of objects: {{degree, institution, graduation_year}})
        - skills (list of strings)
        - certifications (list of strings)  # Fixed: was "certification" in original
        - total_years_experience (integer)

        Provide the output in the following format:
        {self.structured_parser.get_format_instructions()}
        Resume:
        {resume_text}
        """
        response = self.llm.invoke(prompt)
        return self.structured_parser.parse(response.content)

    def infer_insights(self, resume_text: str) -> InferredProfileInsights:
        """Uses LLM to infer high-level insights."""
        prompt = f"""
        Analyze the resume and infer insights based on the professional journey, roles, and skills. Your task is to infer high-level insights about the candidate's profile.
        Extract **only** the following fields:
        - domain (string, e.g., "Data Science")
        - industry (string, e.g., "Healthcare")
        - seniority_level (string, e.g., "Mid-Level")
        - role_type (string, e.g., "Individual Contributor")
        - skills (list of strings, e.g., ["Python", "Machine Learning"])
        - personality_traits (list of strings, e.g., ["Analytical", "Team Player"])
        - workplace_likes (list of strings, e.g., ["Remote Work", "Flexible Hours"])
        - workplace_dislikes (list of strings, e.g., ["Micromanagement", "Long Commutes"])
       
         Provide the output in the following format:
        {self.insight_parser.get_format_instructions()}
        Resume:
        {resume_text}
        """
        response = self.llm.invoke(prompt)
        return self.insight_parser.parse(response.content)

    def analyze(self, file_path: str) -> dict:
        """End-to-end analysis pipeline."""
        resume_text = self.parse_pdf(file_path)
        structured = self.extract_structured_info(resume_text)
        inferred = self.infer_insights(resume_text)
        return {
            "structured_info": structured.model_dump(),
            "inferred_insights": inferred.model_dump()
        }

if __name__ == "__main__":
    analyzer = ResumeAnalyzer(model_name="deepseek/deepseek-chat-v3-0324:free")  # Use correct model name
    file_path = "/Users/vasanthagullapalli/Downloads/Profile.pdf"
    result = analyzer.analyze(file_path)
    print(result)