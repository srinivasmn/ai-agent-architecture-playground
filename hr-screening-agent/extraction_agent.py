from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from schemas import CandidateProfile


# ------------------------------------------------------------------
# LLM INITIALIZATION (DETERMINISTIC)
# ------------------------------------------------------------------

llm = ChatOllama(
    model="llama3",
    temperature=0
)


# ------------------------------------------------------------------
# OUTPUT PARSER (STRUCTURED CONTRACT)
# ------------------------------------------------------------------

parser = PydanticOutputParser(pydantic_object=CandidateProfile)


# ------------------------------------------------------------------
# EXTRACTION PROMPT (LLM RESPONSIBILITY ONLY)
# ------------------------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a data extraction engine.

Your output MUST be valid JSON and NOTHING ELSE.

STRICT RULES:
- Output ONLY a JSON object
- Do NOT add any text before or after the JSON
- Do NOT add explanations
- Do NOT use markdown
- Do NOT rename keys
- Use null for missing optional fields

The JSON object MUST follow this structure exactly:
{{
  "name": string,
  "email": string | null,
  "roles": [ [string, string] ],
  "skills": [string],
  "primary_technologies": [string],
  "current_role": string,
  "education": string,
  "notable_achievements": [string] | null
}}

IMPORTANT:
- "roles" MUST be job DATE RANGES in the form:
  [ ["2019", "Present"], ["2017", "2019"] ]
- Do NOT include job titles or company names in roles

Resume:
{resume_text}
""")


# ------------------------------------------------------------------
# BUILD EXTRACTION CHAIN
# ------------------------------------------------------------------

chain = prompt | llm | parser


# ------------------------------------------------------------------
# AGENT INTERFACE (EXPLICIT CONTRACT)
# ------------------------------------------------------------------

def extract_candidate_profile(resume_text: str) -> CandidateProfile:
    """
    Agent responsibility:
    - Convert unstructured resume text into structured CandidateProfile
    - Enforce schema validity
    - No business logic, no scoring, no reasoning
    """
    return chain.invoke({
        "resume_text": resume_text,
        "format_instructions": parser.get_format_instructions()
    })


def extract_job_requirements(job_text: str):
    """
    Placeholder for Stage 1 JD extraction agent.
    Will be implemented in a dedicated job_extraction_agent module.
    """
    raise NotImplementedError("Job extraction agent not wired yet.")


# ------------------------------------------------------------------
# DETERMINISTIC POST-PROCESSING (NON-LLM LOGIC)
# ------------------------------------------------------------------

def calculate_total_experience(roles: list[list[str]]) -> int:
    """
    Calculate total experience in years from extracted role date ranges.
    This must NOT be done by the LLM.
    """
    current_year = datetime.now().year
    total_years = 0

    for start, end in roles:
        start_year = int(start)
        end_year = current_year if end.lower() == "present" else int(end)
        total_years += end_year - start_year

    return total_years
