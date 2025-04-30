import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from models import StructuredResumeInfo, InferredProfileInsights
from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent

api_key = os.getenv("OPENROUTER_API_KEY")

class LinkedInProfileAnalyzer:
    def __init__(self, model_name="deepseek/deepseek-chat-v3-0324:free", temperature=0):
        # Configure OpenRouter
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://rewireme.me",
                "X-Title": "LinkedIn Profile Analyzer"
            }
        )
        self.structured_parser = PydanticOutputParser(pydantic_object=StructuredResumeInfo)
        self.insight_parser = PydanticOutputParser(pydantic_object=InferredProfileInsights)
        self.ua = UserAgent()

    def scrape_linkedin_profile(self, profile_url: str) -> Optional[str]:
        """Scrapes basic information from a LinkedIn profile."""
        try:
            headers = {
                'User-Agent': self.ua.random,
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            response = requests.get(profile_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract basic profile information
            name = soup.find('h1', class_='text-heading-xlarge').get_text(strip=True) if soup.find('h1', class_='text-heading-xlarge') else "N/A"
            headline = soup.find('div', class_='text-body-medium').get_text(strip=True) if soup.find('div', class_='text-body-medium') else "N/A"
            
            # Extract about section
            about_section = soup.find('div', {'id': 'about'})
            about = about_section.find_next('div').get_text(strip=True) if about_section else "N/A"
            
            # Extract experience
            experience_section = soup.find('section', {'id': 'experience-section'})
            experience = []
            if experience_section:
                for item in experience_section.find_all('li', class_='artdeco-list__item'):
                    title = item.find('h3', class_='t-16').get_text(strip=True) if item.find('h3', class_='t-16') else "N/A"
                    company = item.find('p', class_='t-14').get_text(strip=True) if item.find('p', class_='t-14') else "N/A"
                    duration = item.find('span', class_='t-14').get_text(strip=True) if item.find('span', class_='t-14') else "N/A"
                    experience.append(f"{title} at {company} ({duration})")
            
            # Extract education
            education_section = soup.find('section', {'id': 'education-section'})
            education = []
            if education_section:
                for item in education_section.find_all('li', class_='artdeco-list__item'):
                    school = item.find('h3', class_='t-16').get_text(strip=True) if item.find('h3', class_='t-16') else "N/A"
                    degree = item.find('p', class_='t-14').get_text(strip=True) if item.find('p', class_='t-14') else "N/A"
                    education.append(f"{degree} from {school}")
            
            # Extract skills
            skills_section = soup.find('section', {'id': 'skills-section'})
            skills = []
            if skills_section:
                for item in skills_section.find_all('span', class_='t-16'):
                    skills.append(item.get_text(strip=True))
            
            # Compile all information
            profile_text = f"""
            Name: {name}
            Headline: {headline}
            
            About:
            {about}
            
            Experience:
            {chr(10).join(experience) if experience else "N/A"}
            
            Education:
            {chr(10).join(education) if education else "N/A"}
            
            Skills:
            {', '.join(skills) if skills else "N/A"}
            """
            
            return profile_text
            
        except Exception as e:
            print(f"Error scraping LinkedIn profile: {e}")
            return None

    def extract_structured_info(self, profile_text: str) -> StructuredResumeInfo:
        """Uses LLM to extract structured profile info."""
        prompt = f"""
        Extract structured information from the following LinkedIn profile. Convert the information into a structured format as per the following instructions:   
        Normalize the job titles based on official published job roles and extract skills from job description.
        Calculate total years of experience based on the work experience section and date ranges for each job.
        Infer the location based on the profile information if available.
        Provide the output in the following format:
        {self.structured_parser.get_format_instructions()}
        
        LinkedIn Profile:
        {profile_text}
        """
        response = self.llm.invoke(prompt)
        return self.structured_parser.parse(response.content)

    def infer_insights(self, profile_text: str) -> InferredProfileInsights:
        """Uses LLM to infer high-level insights."""
        prompt = f"""
        Analyze the LinkedIn profile and provide insights:
        Infer the location based on the profile information if available.
        Infer the domain and industry based on the work experience section.
        Infer the experience level based on the job titles and responsibilities.
        Infer the role type (individual contributor or manager) based on the job titles and responsibilities.
        Infer personality traits, workplace likes/dislikes based on tenure in each role/company and career progression. 
        Analyze the "About" section for additional personality and professional traits.
        {self.insight_parser.get_format_instructions()}
        
        LinkedIn Profile:
        {profile_text}
        """
        response = self.llm.invoke(prompt)
        return self.insight_parser.parse(response.content)

    def analyze(self, profile_url: str) -> dict:
        """End-to-end analysis pipeline."""
        profile_text = self.scrape_linkedin_profile(profile_url)
        if not profile_text:
            raise ValueError("Failed to scrape LinkedIn profile")
            
        structured = self.extract_structured_info(profile_text)
        inferred = self.infer_insights(profile_text)
        return {
            "structured_info": structured.model_dump(),
            "inferred_insights": inferred.model_dump()
        }

if __name__ == "__main__":
    analyzer = LinkedInProfileAnalyzer(model_name="deepseek/deepseek-chat-v3-0324:free")
    profile_url = "https://www.linkedin.com/in/vgulla"  # Replace with actual profile URL
    result = analyzer.analyze(profile_url)
    print(result)