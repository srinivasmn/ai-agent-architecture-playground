from pydantic import BaseModel
from typing import List, Optional, Tuple


class CandidateProfile(BaseModel):
    # -----------------------------
    # Basic identity
    # -----------------------------
    name: str
    email: Optional[str] = None

    # -----------------------------
    # Work history
    # Each tuple is (start_year, end_year or "Present")
    # Example: [("2019", "Present"), ("2017", "2019")]
    # -----------------------------
    roles: List[Tuple[str, str]]

    # -----------------------------
    # Technical skills
    # -----------------------------
    skills: List[str]
    primary_technologies: List[str]

    # -----------------------------
    # Current status
    # -----------------------------
    current_role: str
    education: str

    # -----------------------------
    # Optional highlights
    # -----------------------------
    notable_achievements: Optional[List[str]] = None


class JobRequirements(BaseModel):
    required_skills: List[str]
    nice_to_have_skills: List[str]
    minimum_experience_years: int