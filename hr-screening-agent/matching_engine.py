from pathlib import Path
from typing import Dict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from schemas import CandidateProfile
from pydantic import BaseModel

from extraction_agent import calculate_total_experience


SKILL_ALIASES = {
    "postgres": "postgresql",
    "postgre sql": "postgresql",
    "rest api": "restful api",
    "rest apis": "restful api",
    "fast api": "fastapi",
    "aws ec2": "aws",
}

def normalize_skill(skill: str) -> str:
    skill = skill.lower().strip()
    return SKILL_ALIASES.get(skill, skill)


def skill_matches(jd_skill: str, resume_skill: str) -> bool:
    jd_tokens = set(jd_skill.split())
    resume_tokens = set(resume_skill.split())
    return len(jd_tokens & resume_tokens) > 0



def compute_skill_match(jd_skills: list[str], resume_skills: list[str]) -> float:
    jd_normalized = [normalize_skill(s) for s in jd_skills]
    resume_normalized = [normalize_skill(s) for s in resume_skills]

    matched = 0
    for jd_skill in jd_normalized:
        if any(skill_matches(jd_skill, res_skill) for res_skill in resume_normalized):
            matched += 1

    return matched / len(jd_normalized) if jd_normalized else 0.0





# ------------------------------------------------------------------
# JOB REQUIREMENTS SCHEMA
# ------------------------------------------------------------------

class JobRequirements(BaseModel):
    required_skills: list[str]
    nice_to_have_skills: list[str]
    minimum_experience_years: int


# ------------------------------------------------------------------
# MATCH RESULT SCHEMA
# ------------------------------------------------------------------

class MatchResult(BaseModel):
    required_skills_match: float
    nice_to_have_match: float
    experience_match: bool
    final_score: int
    decision: str
    reason: str


# ------------------------------------------------------------------
# LLM (ONLY USED FOR JD EXTRACTION)
# ------------------------------------------------------------------

llm = ChatOllama(model="phi3", temperature=0)

jd_parser = PydanticOutputParser(pydantic_object=JobRequirements)

jd_prompt = ChatPromptTemplate.from_template("""
You are a data extraction engine.

Your output MUST be valid JSON and NOTHING ELSE.

STRICT RULES:
- Output ONLY a JSON object
- Do NOT add any text before or after the JSON
- Do NOT add explanations
- Do NOT use markdown
- Do NOT use bullet points
- Do NOT use headings
- Do NOT rename keys
- Use empty lists if no skills are found

The JSON object MUST follow this structure exactly:
{{
  "required_skills": [string],
  "nice_to_have_skills": [string],
  "minimum_experience_years": number
}}

IMPORTANT CLARIFICATIONS:
- required_skills = skills that are explicitly mandatory
- nice_to_have_skills = skills listed as optional or nice-to-have
- minimum_experience_years = numeric value only (example: 5)

Job Description:
{job_description}
""")


jd_chain = jd_prompt | llm | jd_parser


# ------------------------------------------------------------------
# AGENT INTERFACE (JOB REQUIREMENTS EXTRACTION)
# ------------------------------------------------------------------

def extract_job_requirements(job_text: str):
    """
    Agent responsibility:
    - Convert unstructured JD into structured JobRequirements
    - Enforce schema validity
    - No scoring, no business logic
    """
    return jd_chain.invoke({
        "job_description": job_text
    })


# ------------------------------------------------------------------
# MATCHING LOGIC (PURE PYTHON)
# ------------------------------------------------------------------

def compute_match(
    candidate: CandidateProfile,
    experience_years: int,
    job: JobRequirements
) -> MatchResult:

    # Required skills (normalized, token-based)
    required_score = compute_skill_match(
        job.required_skills,
        candidate.skills
    )

    # Nice-to-have skills (normalized, token-based)
    nice_score = compute_skill_match(
        job.nice_to_have_skills,
        candidate.skills
    )



    # Experience check
    experience_ok = experience_years >= job.minimum_experience_years

    # Weighted scoring
    final_score = int(
        (required_score * 60) +
        (nice_score * 20) +
        ((1 if experience_ok else 0) * 20)
    )

    decision = "Accept" if final_score >= 70 else "Reject"

    reason = (
        "Strong overall match"
        if decision == "Accept"
        else "Meets experience but lacks core backend requirements"
    )

    return MatchResult(
        required_skills_match=round(required_score, 2),
        nice_to_have_match=round(nice_score, 2),
        experience_match=experience_ok,
        final_score=final_score,
        decision=decision,
        reason=reason
    )


# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent

    # Load JD
    with open(BASE_DIR / "job_description.txt", "r", encoding="utf-8") as f:
        job_description = f.read()

    job_requirements = jd_chain.invoke({
        "job_description": job_description
    })

    # Load candidate (reuse Stage 1)
    from stage1_extraction import chain as resume_chain

    with open(BASE_DIR / "resume_2.txt", "r", encoding="utf-8") as f:
        resume_text = f.read()

    candidate = resume_chain.invoke({
        "resume_text": resume_text
    })

    experience_years = calculate_total_experience(candidate.roles)

    match_result = compute_match(candidate, experience_years, job_requirements)

    print("\nMATCH RESULT")
    print("=" * 60)
    print(match_result.model_dump())
