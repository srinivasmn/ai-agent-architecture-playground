"""
Pipeline Orchestrator for AI-Powered Resume Screening Agent

Stages:
1. Extraction Agent (LLM -> Structured Schema)
2. Normalization & Scoring Engine (Deterministic Policy Layer)
3. Decision & Explanation

This file represents the control plane of the agentic system.
"""

from pathlib import Path

from extraction_agent import extract_candidate_profile, calculate_total_experience
from matching_engine import compute_match, extract_job_requirements
from schemas import CandidateProfile, JobRequirements


BASE_DIR = Path(__file__).parent


def load_job_description() -> str:
    with open(BASE_DIR / "data" / "job_description.txt", "r", encoding="utf-8") as f:
        return f.read()


def load_resumes() -> list[str]:
    resumes_dir = BASE_DIR / "data" / "resumes"
    resumes = []
    for resume_file in sorted(resumes_dir.glob("resume_*.txt")):
        with open(resume_file, "r", encoding="utf-8") as f:
            resumes.append(f.read())
    return resumes


def run_screening_pipeline():
    print("\n[PIPELINE] Starting HR Screening Agent\n")

    # -------------------------------
    # Stage 1: Job Understanding
    # -------------------------------
    print("[STAGE 1] Extracting job requirements...")
    job_text = load_job_description()
    job_requirements: JobRequirements = extract_job_requirements(job_text)

    # -------------------------------
    # Stage 1: Candidate Extraction
    # -------------------------------
    print("[STAGE 1] Extracting candidate profiles...")
    resumes = load_resumes()
    candidates: list[CandidateProfile] = []

    for resume_text in resumes:
        candidate = extract_candidate_profile(resume_text)
        candidates.append(candidate)

    # -------------------------------
    # Stage 2: Matching & Scoring
    # -------------------------------
    print("[STAGE 2] Evaluating candidates...\n")

    results = []
    for candidate in candidates:
        experience_years = calculate_total_experience(candidate.roles)
        match_result = compute_match(candidate, experience_years, job_requirements)
        results.append((candidate.name, match_result))

    # -------------------------------
    # Decision & Explanation
    # -------------------------------
    print("\nFINAL RANKED RESULTS")
    print("=" * 60)

    ranked = sorted(results, key=lambda x: x[1].final_score, reverse=True)

    for rank, (name, result) in enumerate(ranked, start=1):
        print(f"\nRank {rank}: {name}")
        print(result.model_dump())


if __name__ == "__main__":
    run_screening_pipeline()
