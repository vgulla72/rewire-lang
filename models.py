from pydantic import BaseModel, EmailStr
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser


class EducationEntry(BaseModel):
    degree: str
    institution: str
    graduation_year: Optional[str]

class ExperienceEntry(BaseModel):
    title: str
    company: str
    duration: Optional[str]

class StructuredResumeInfo(BaseModel):
    full_name: Optional[str]
    email: Optional[EmailStr]
    phone_number: Optional[str]
    education: List[EducationEntry]
    work_experience: List[ExperienceEntry]
    skills: List[str]
    certifications: List[str]
    total_years_experience: Optional[int]

class InferredProfileInsights(BaseModel):
    domain: str  # e.g., "Data Science"
    industry: str  # e.g., "Healthcare"
    experience_level: str  # e.g., "Senior"
    compensation_range_usd: str  # e.g., "$120,000 - $140,000"
    role_type: str  # e.g., "Individual Contributor"
    personality_traits: List[str]  # e.g., ["Analytical", "Team Player"]



