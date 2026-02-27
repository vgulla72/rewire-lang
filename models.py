from pydantic import BaseModel, EmailStr
from typing import List, Optional, Literal
from langchain_core.output_parsers import PydanticOutputParser


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
    certifications: Optional[List[Optional[str]]] = []
    total_years_experience: Optional[int]

class InferredProfileInsights(BaseModel):
    domain: Optional[str]  # e.g., "Data Science"
    industry: Optional[str]  # e.g., "Healthcare"
    seniority_level: Optional[str]  # e.g., "Mid-Level"
    role_type: Optional[str]  # e.g., "Individual Contributor"
    skills: Optional[List[Optional[str]]] = []  # e.g., ["Python", "Machine Learning"]
    personality_traits: Optional[List[Optional[str]]] = []  # e.g., ["Analytical", "Team Player"]
    workplace_likes: Optional[List[Optional[str]]] = []  # e.g., ["Remote Work", "Flexible Hours"]
    workplace_dislikes: Optional[List[Optional[str]]] = []  # e.g., ["Micromanagement", "Long Commutes"]

class question_answer(BaseModel):
    question: str
    # A list of multiple choice options. The correct answer should be one of these.
    options: List[str]
    # The correct answer text (should match one of the options)
    answer: str

class question_input(BaseModel):
    topic: str
    age: Optional[int] = None
    gender: Optional[str] = None

    

# Pydantic model to structure input
class CareerInput(BaseModel):
    structured_info: dict
    inferred_insights: dict
    career_change_reason: str
    hobbies_and_passions: str
    preferred_engagement: str
    compensation_preference: Optional[str] = None  # e.g., "Flexible"

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