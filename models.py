from pydantic import BaseModel, EmailStr
from typing import List, Optional, Literal
from langchain.output_parsers import PydanticOutputParser


class EducationEntry(BaseModel):
    degree: Optional[str] = None
    institution: str
    graduation_year: Optional[str] = None  # e.g., "2020", "Expected 2025"

class ExperienceEntry(BaseModel):
    title: str
    company: str
    duration: Optional[str] = None

class StructuredResumeInfo(BaseModel):
    full_name: Optional[str]
    email: Optional[EmailStr]
    phone_number: Optional[str]
    education: List[EducationEntry]
    location: Optional[str] # e.g., "Greater Seattle Area, WA"
    work_experience: List[ExperienceEntry]
    certifications: List[str]
    total_years_experience: Optional[int]

class InferredProfileInsights(BaseModel):
    domain: str  # e.g., "Data Science"
    industry: str  # e.g., "Healthcare"
    seniority_level: str  # e.g., "Mid-Level"
    role_type: str  # e.g., "Individual Contributor"
    skills: List[str]  # e.g., ["Python", "Machine Learning"]
    personality_traits: List[str]  # e.g., ["Analytical", "Team Player"]
    workplace_likes: List[str]  # e.g., ["Remote Work", "Flexible Hours"]
    workplace_dislikes: List[str]  # e.g., ["Micromanagement", "Long Commutes"]

# Pydantic model to structure input
class CareerInput(BaseModel):
    structured_info: dict
    inferred_insights: dict
    career_change_reason: str
    hobbies_and_passions: str

# Define output model
class CareerRecommendation(BaseModel):
    title: str
    alternative_titles: List[str]  # e.g., ["Technology Consultant", "Startup Advisor"]
    reason: str
    compensation_range: Optional[str]  # e.g., "$80,000 - $120,000"
    trending_skills: List[str]
    suggested_training: List[str]
    preparation_steps: List[str]

class CareerRecommendationsOutput(BaseModel):
    career_recommendations: List[CareerRecommendation]

# Define input model
class CompanyInput(BaseModel):
    structured_info: dict
    inferred_insights: dict
    career_change_reason: str
    hobbies_and_passions: str
    career_recommendations: CareerRecommendationsOutput

# Define output model
class CompanyRecommendation(BaseModel):
    company: str
    category: str  # e.g., Non-Profit, For-Profit, Academia, Government
    reason: str

class CompanyRecommendationsOutput(BaseModel):
    recommendations: List[CompanyRecommendation]

# Input Model
class PeopleSearchInput(BaseModel):
    previous_title: str
    location: str
    recommended_roles: CareerRecommendationsOutput
    recommended_companies: CompanyRecommendationsOutput

# -----------------------------
# Output Models
# -----------------------------
class PersonExample(BaseModel):
    name: str
    previous_title: str
    current_title: str
    current_company: str
    linkedin_profile: str
    summary: str

class PeopleSearchOutput(BaseModel):
    matches: List[PersonExample]

class sectorrecommendation(BaseModel):
    sector: str
    reason: str

class SectorRecommendationsOutput(BaseModel):
    sectorrecommendations: List[sectorrecommendation]