import os
import pdfplumber
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from models import StructuredResumeInfo, InferredProfileInsights  # your Pydantic models

class ResumeAnalyzer:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        os.environ["OPENAI_MODEL_NAME"] = model_name
        self.llm = ChatOpenAI(temperature=temperature)
        self.structured_parser = PydanticOutputParser(pydantic_object=StructuredResumeInfo)
        self.insight_parser = PydanticOutputParser(pydantic_object=InferredProfileInsights)

    def parse_pdf(self, file_path: str) -> str:
        """Extracts text content from a PDF resume."""
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text

    def extract_structured_info(self, resume_text: str) -> StructuredResumeInfo:
        """Uses LLM to extract structured resume info (e.g., name, roles, skills)."""
        prompt = f"""
        Extract structured information from the following resume.
        Format the output as JSON in this schema:
        {self.structured_parser.get_format_instructions()}

        Resume:
        {resume_text}
        """
        response = self.llm.invoke(prompt)
        return self.structured_parser.parse(response.content)

    def infer_insights(self, resume_text: str) -> InferredProfileInsights:
        """Uses LLM to infer high-level insights from the resume text."""
        prompt = f"""
        Based on the resume text below, infer:
        - Primary Domain (e.g., Data Science, Frontend Engineering)
        - Likely Industry (e.g., Fintech, E-commerce)
        - Experience Level (e.g., Entry, Mid, Senior, Executive)
        - Estimated Compensation Range in USD (e.g., 100k-120k)
        - Individual Contributor or Managerial Role
        - Likely Personality Traits

        Return the result as JSON in this format:
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


# Example
if __name__ == "__main__":
    analyzer = ResumeAnalyzer()
    file_path = "/Users/vasanthagullapalli/Documents/Vasantha Gullapalli Resume.pdf"
    result = analyzer.analyze(file_path)
    print(result)
