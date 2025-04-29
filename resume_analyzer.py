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
        Extract structured information from the following resume.
        {self.structured_parser.get_format_instructions()}
        Resume:
        {resume_text}
        """
        response = self.llm.invoke(prompt)
        return self.structured_parser.parse(response.content)

    def infer_insights(self, resume_text: str) -> InferredProfileInsights:
        """Uses LLM to infer high-level insights."""
        prompt = f"""
        Analyze the resume and provide insights:
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
    file_path = "/Users/vasanthagullapalli/Documents/Vasantha Gullapalli Resume.pdf"
    result = analyzer.analyze(file_path)
    print(result)