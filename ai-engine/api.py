# ================= IMPORTS =================
import os
import builtins
import pydantic
# Hack to fix broken Cashfree SDK v4.1.2 which forgets to import StrictBytes
if not hasattr(builtins, "StrictBytes"):
    builtins.StrictBytes = getattr(pydantic, "StrictBytes", str) # Fallback to str if missing

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Set
import pdfplumber
import numpy as np
import json
import math
import re
import os
import hmac
import hashlib
import requests
import time
from functools import lru_cache
from io import BytesIO
from huggingface_hub import InferenceClient
import resend
from cashfree_pg.models.create_order_request import CreateOrderRequest
from cashfree_pg.api_client import Cashfree
from cashfree_pg.models.customer_details import CustomerDetails
from cashfree_pg.models.order_meta import OrderMeta
import asyncio
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import os
from groq import Groq
import google.generativeai as genai

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Initialize AI Clients
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
else:
    gemini_model = None

print(f"[Startup] Groq Initialized: {bool(GROQ_API_KEY)}")
print(f"[Startup] Gemini Initialized: {bool(GOOGLE_API_KEY)}")
print(f"[Startup] Loading .env from: {env_path}")

# ================= APP =================

app = FastAPI()

# ================= AI RESPONSE CACHE =================
# Simple in-memory cache to prevent redundant LLM calls for the same payload
AI_RESPONSE_CACHE = {}

def get_cache_key_backend(prefix: str, **kwargs):
    # Create a stable key from the input arguments
    items = sorted(kwargs.items())
    key_base = f"{prefix}:{str(items)}"
    return hashlib_mdsafe_hex(key_base)

def hashlib_mdsafe_hex(s: str):
    return hashlib.md5(s.encode()).hexdigest()

# ================= CORS CONFIGURATION =================
# Allow both localhost (dev) and production domains

FRONTEND_URL = os.getenv("FRONTEND_URL")

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://asterix-jobs.vercel.app",
    "https://www.asterix-jobs.vercel.app",
    "https://asterix-jobs.in",
    "https://www.asterix-jobs.in",
]

if FRONTEND_URL:
    ALLOWED_ORIGINS.append(FRONTEND_URL)

# filter empty values BEFORE adding middleware
ALLOWED_ORIGINS = [o for o in ALLOWED_ORIGINS if o]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= LIGHTWEIGHT EMBEDDER =================

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

class HFEmbedder:
    """Cloud-based neural embedder via Hugging Face InferenceClient"""
    
    def __init__(self, api_key: str):
        self.client = InferenceClient(token=api_key)
        self.model_id = HF_MODEL_ID
        self.dim = 384 # all-MiniLM-L6-v2 standard
        self._executor = ThreadPoolExecutor(max_workers=10)
        print(f"[Neural] InferenceClient Initialized for: {self.model_id}")

    @lru_cache(maxsize=2048)
    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Fetch dense vectors using InferenceClient with auto-retries"""
        try:
            # InferenceClient handles retries and 503s internally
            vec_list = self.client.feature_extraction(
                text[:1500],
                model=self.model_id
            )
            
            vec = np.array(vec_list, dtype=np.float32)
            
            # Handle nested lists if returned
            if len(vec.shape) > 1:
                vec = np.mean(vec, axis=0)
            
            if normalize and np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            return vec
            
        except Exception as e:
            print(f"[Neural ERROR] InferenceClient failed: {e}")
            return np.zeros(self.dim, dtype=np.float32)

@app.on_event("startup")
def initialize():
    global embedder
    embedder = HFEmbedder(api_key=HF_API_KEY)
    global job_cache
    job_cache = {} # Job text hash -> Embedding vector
    print("[Startup] Cloud Neural Engine initialized")
    print(f"[Startup] Job Cache initialized")
    print(f"[Startup] CORS Allowed Origins: {ALLOWED_ORIGINS}")

# ================= EMAIL CONFIGURATION =================

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
print(f"[Startup] RESEND_API_KEY loaded: {bool(RESEND_API_KEY)}")
if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY
    print("[Email] Resend client initialized")
else:
    print("[Email Warning] RESEND_API_KEY not found in environment")


# ================= CASHFREE CONFIGURATION =================

CASHFREE_APP_ID = os.getenv("CASHFREE_APP_ID")
CASHFREE_SECRET_KEY = os.getenv("CASHFREE_SECRET_KEY")

if CASHFREE_APP_ID and CASHFREE_SECRET_KEY:
    Cashfree.XClientId = CASHFREE_APP_ID
    Cashfree.XClientSecret = CASHFREE_SECRET_KEY
    cf_env = os.getenv("CASHFREE_ENV", "sandbox").lower()
    if cf_env == "production":
        Cashfree.XEnvironment = Cashfree.PRODUCTION
        print("[Cashfree] Client initialized (Production mode)")
    else:
        Cashfree.XEnvironment = Cashfree.SANDBOX
        print("[Cashfree] Client initialized (Sandbox mode)")
else:
    print("[Cashfree Warning] CASHFREE_APP_ID or CASHFREE_SECRET_KEY not found")


# ================= PDF EXTRACTION =================

def extract_text_from_pdf(file: UploadFile) -> str:
    """Robust PDF text extraction with fallbacks"""
    try:
        file.file.seek(0)
        content = file.file.read()
        
        print(f"[PDF] File: {file.filename}, Size: {len(content)} bytes")
        
        with pdfplumber.open(BytesIO(content)) as pdf:
            total_pages = len(pdf.pages)
            print(f"[PDF] Pages: {total_pages}")
            
            text_parts = []
            
            for i, page in enumerate(pdf.pages[:3]):  # Max 3 pages
                # Try text extraction
                page_text = page.extract_text()
                
                if not page_text or len(page_text.strip()) < 50:
                    # Fallback: extract from tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                text_parts.append(" ".join([str(cell) for cell in row if cell]))
                else:
                    text_parts.append(page_text)
                
                print(f"[PDF] Page {i+1}: {len(page_text or '')} chars extracted")
                
                # Stop if we have enough content
                combined = " ".join(text_parts)
                if len(combined) > 2500:
                    break
            
            final_text = " ".join(text_parts).strip()
            final_text = re.sub(r'\s+', ' ', final_text)  # Normalize whitespace
            final_text = final_text[:3000]  # Hard limit
            
            print(f"[PDF] Final length: {len(final_text)} chars")
            print(f"[PDF] Preview: {final_text[:150]}...")
            
            return final_text
            
    except Exception as e:
        print(f"[PDF ERROR] {type(e).__name__}: {str(e)}")
        return ""


def extract_content(file: UploadFile) -> str:
    """Helper to extract text or fallback if needed"""
    # If it's a PDF, use pdfplumber
    if file.filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file)
    
    # If it's a text file, read directly
    try:
        file.file.seek(0)
        content = file.file.read()
        return content.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[EXTRACT ERROR] {file.filename}: {e}")
        return ""


# ================= DOCUMENT VALIDATION =================

def is_authentic_resume(text: str) -> bool:
    """Heuristic check to see if a document is actually a professional resume"""
    if not text or len(text) < 300:
        return False
        
    text_lower = text.lower()
    
    # Positive markers: Essential resume sections
    # Resumes almost always contain at least 2 of these
    POSITIVE_MARKERS = [
        "experience", "education", "skills", "projects", "work history",
        "employment", "achievements", "summary", "objective", "certifications",
        "university", "college", "institue", "bachelor", "master", "phd"
    ]
    
    # Negative markers: Indicators of non-resume documents (e.g. academic papers)
    # If these are highly dominant, it might be a false positive
    NEGATIVE_MARKERS = [
        "abstract", "introduction", "methodology", "conclusion", "references",
        "figure 1", "table 1", "et al.", "1st class", "paper code", "roll no"
    ]
    
    positive_hits = sum(1 for m in POSITIVE_MARKERS if m in text_lower)
    negative_hits = sum(1 for m in NEGATIVE_MARKERS if m in text_lower)
    
    # Guard logic:
    # 1. Must have at least 2 professional/educational sections
    # 2. Rejection markers must not outweigh positive markers significantly
    if positive_hits < 2:
        print(f"[Guard] LOW POSITIVE MARKERS: {positive_hits}. Likely not a resume.")
        return False
        
    if negative_hits > positive_hits:
        print(f"[Guard] HIGH NEGATIVE MARKERS: {negative_hits} vs {positive_hits}. Likely a paper/doc.")
        return False
        
    return True


# ================= TEXT PROCESSING =================

def tokenize(text: str) -> Set[str]:
    """Extract meaningful tokens from text"""
    STOPWORDS = {
        "the", "and", "for", "with", "this", "that", "are", "was",
        "you", "will", "have", "from", "your", "not", "can", "but",
        "work", "role", "team", "job", "been", "what", "which", "also",
        "more", "their", "into", "through", "about", "other"
    }
    
    # Extract alphanumeric words, keeping special chars like C++, .NET
    words = re.findall(r'\b[a-z][a-z0-9+#.\-]{1,}\b', text.lower())
    tokens = {w for w in words if len(w) >= 2 and w not in STOPWORDS}
    
    return tokens


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between vectors"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


# ================= SCORING FUNCTIONS =================

def detect_soft_skill_environment(job_text: str) -> bool:
    """Detect if the job primarily values soft skills (communication, leadership, etc) over pure technical skills"""
    SOFT_SKILL_KEYWORDS = [
        "communication", "interpersonal", "leadership", "teamwork", "adaptability",
        "problem-solving", "critical thinking", "collaboration", "client-facing",
        "presentation", "negotiation", "empathy", "talking", "public speaking",
        "stakeholder management", "relationship building", "conflict resolution"
    ]
    TECH_SKILL_KEYWORDS = [
        "python", "javascript", "typescript", "java", "c++", "c#", "rust", "golang",
        "fullstack", "frontend", "backend", "devops", "cloud", "aws", "azure", "gcp",
        "docker", "kubernetes", "sql", "react", "node", "html", "css", "web development"
    ]
    
    jt_lower = job_text.lower()
    soft_hits = sum(1 for k in SOFT_SKILL_KEYWORDS if k in jt_lower)
    tech_hits = sum(1 for k in TECH_SKILL_KEYWORDS if k in jt_lower)
    
    # Environment is considered soft-skill heavy if soft skills outnumber tech skills 
    # OR if there are a significant number of soft skills present.
    return soft_hits >= 3 or (soft_hits > 0 and soft_hits >= tech_hits)

def compute_semantic_score(text: str, job_text: str, min_ratio: float = 0.07) -> float:
    """Compute semantic similarity between text and job"""
    
    if not text or len(text) < 30:
        print(f"[Score] Text too short: {len(text)} chars")
        return 0.0
    
    # Step 1: Token overlap (fast initial filter)
    text_tokens = tokenize(text)
    job_tokens = tokenize(job_text)
    
    overlap_count = len(text_tokens & job_tokens)
    job_token_count = len(job_tokens)
    
    if job_token_count == 0:
        return 0.0
    
    keyword_ratio = overlap_count / job_token_count
    
    print(f"[Score] Tokens: {len(text_tokens)} text, {job_token_count} job")
    print(f"[Score] Overlap: {overlap_count} tokens ({keyword_ratio:.1%})")
    
    # If very low overlap, skip expensive embedding computation
    if keyword_ratio < min_ratio:
        print(f"[Score] Keyword ratio too low ({keyword_ratio:.1%}/{min_ratio:.1%}), returning 0")
        return 0.0
    
    # This function is now intended to be called via asyncio.to_thread or similar
    # if high concurrency is needed, but we'll optimize the calling side.
    
    # Step 2: Embedding similarity
    try:
        # Check cache for job embedding
        job_hash = hash(job_text[:2000])
        if job_hash in job_cache:
            job_vec = job_cache[job_hash]
            print("[Score] Using cached job embedding")
        else:
            job_vec = embedder.encode(job_text[:2000], normalize=True)
            job_cache[job_hash] = job_vec
            
        text_vec = embedder.encode(text[:2000], normalize=True)
        
        cosine = cosine_similarity(text_vec, job_vec)
        print(f"[Score] Cosine similarity: {cosine:.3f}")
        
        # Normalize: typical range is 0.1-0.5 for real matches
        # Normalize: broaden the range for better distribution
        BASELINE = 0.18 # Higher baseline to filter noise
        CEILING = 0.75  # Higher ceiling
        
        if cosine < BASELINE:
            return 0.0
        
        normalized = (cosine - BASELINE) / (CEILING - BASELINE)
        final_score = min(1.0, max(0.0, normalized))
        
        print(f"[Score] Normalized score: {final_score:.3f}")
        return final_score
        
    except Exception as e:
        print(f"[Score ERROR] {e}")
        return keyword_ratio * 0.5  # Fallback to keyword score


def compute_skill_match(skills: List[dict], job_text: str) -> float:
    """
    Coverage-based skill matching with MUST-HAVE penalty.
    """
    job_tokens = tokenize(job_text)
    job_text_lower = job_text.lower()
    matched_weight = 0
    matched_skills = []
    
    # Target saturation: ~35-40 points for 100% score.
    TARGET_SATURATION = 40 
    
    for skill_obj in skills:
        skill_name = (skill_obj.get("skill") or "").strip().lower()
        weight = max(int(skill_obj.get("weight") or 5), 5) 
        
        if not skill_name:
            continue
            
        skill_tokens = tokenize(skill_name)
        
        if skill_tokens & job_tokens:
            matched_weight += weight
            matched_skills.append(skill_name)
    
    # Calculate coverage score
    coverage_ratio = min(matched_weight / TARGET_SATURATION, 1.0)
    
    # --- STRATEGIC PENALTY SYSTEM ---
    # Identify explicit "Required" keywords in the JD
    # If these are missing, we apply a significant multiplier penalty
    REQUIRED_KEYWORDS = ["azure", "aws", "gcp", "devops", "kubernetes", "docker", "cloud", "terraform", "jenkins", "ci/cd"]
    missing_critical = []
    
    for kw in REQUIRED_KEYWORDS:
        # Check if the JD mentions it as a requirement (simple heuristic)
        if re.search(r'\b' + re.escape(kw) + r'\b', job_text_lower):
            # Check if candidate has it
            candidate_has = any(kw in (s.get("skill") or "").lower() for s in skills) or \
                            any(kw in t for t in matched_skills)
            
            if not candidate_has:
                missing_critical.append(kw)
    
    # Penalty multiplier: 0.8^n where n is the number of missing critical skills
    # This prevents total skill gaps from getting high scores
    penalty = 1.0
    if missing_critical:
        penalty = 0.8 ** min(len(missing_critical), 4)
        print(f"[Skills] MISSING CRITICAL: {missing_critical}, Penalty: {penalty:.2f}")
    
    final_skill_score = coverage_ratio * penalty
    
    print(f"[Skills] Matched: {matched_skills[:5]}")
    print(f"[Skills] Final Score: {final_skill_score:.2%} (Coverage={coverage_ratio:.1%}, Penalty={penalty:.2f})")
    
    return final_skill_score


def compute_profile_quality(profile_text: str, skills: List[dict]) -> float:
    """Assess profile completeness"""
    score = 0.0
    
    # Content length check (40%)
    meaningful_words = [w for w in profile_text.split() if len(w) > 3]
    word_score = min(len(meaningful_words) / 40.0, 1.0)
    score += word_score * 0.4
    
    # Skill count check (40%)
    skill_score = min(len(skills) / 5.0, 1.0)
    score += skill_score * 0.4
    
    # Experience indicators (20%)
    exp_markers = [" at ", " in ", "years", "experience", "worked"]
    has_exp = any(marker in profile_text.lower() for marker in exp_markers)
    score += 0.2 if has_exp else 0.0
    
    print(f"[Profile] Quality: {score:.1%} (words={word_score:.1%}, skills={skill_score:.1%}, exp={has_exp})")
    
    return max(0.3, min(1.0, score))


def compute_skill_mastery(
    skill: str,
    resume_tokens: Set[str],
    job_tokens: Set[str],
    profile_tokens: Set[str]
) -> int:
    """Score individual skill mastery"""
    skill_tokens = tokenize(skill)
    
    if not skill_tokens:
        return 25
    
    # Calculate overlap ratios
    resume_match = len(skill_tokens & resume_tokens) / len(skill_tokens)
    job_match = len(skill_tokens & job_tokens) / len(skill_tokens)
    profile_match = len(skill_tokens & profile_tokens) / len(skill_tokens)
    
    # Weighted combination
    base_score = (
        0.50 * resume_match +
        0.30 * job_match +
        0.20 * profile_match
    )
    
    # Add deterministic variance for realism
    variance = ((hash(skill) % 30) - 15) / 100.0
    final = base_score + variance
    
    return max(10, min(100, round(final * 100)))


def generate_highlights(
    resume_tokens: Set[str],
    job_tokens: Set[str],
    skills: List[dict],
    profile_text: str,
    score: int
) -> List[str]:
    """Generate contextual match insights"""
    highlights = []
    
    # Find matched skills (Audit the full list, not just top 8)
    matched_skills = []
    for skill_obj in skills:
        skill_name = (skill_obj.get("skill") or "").strip().lower()
        if skill_name and tokenize(skill_name) & job_tokens:
            matched_skills.append(skill_name.title())
    
    if matched_skills:
        # Show up to 6 matches for better transparency
        highlights.append(f"Matched skills: {', '.join(matched_skills[:6])}")
    
    # Experience check
    has_exp = any(m in profile_text.lower() for m in [" at ", "years", "experience"])
    if has_exp:
        highlights.append("Relevant experience found in profile")
    
    # Score-based insight
    if score >= 75:
        highlights.append("Excellent alignment with role requirements")
    elif score >= 60:
        highlights.append("Strong match for this position")
    elif score >= 40:
        highlights.append("Moderate fit with development areas")
    else:
        highlights.append("Skills gap identified - focus on key requirements")
    
    # Token overlap insight
    overlap = len(resume_tokens & job_tokens)
    if overlap > 20:
        highlights.append("High technical vocabulary match")
    elif overlap > 10:
        highlights.append("Good keyword alignment")
    
    return highlights[:3]


# ================= HEALTH =================

@app.get("/")
def health():
    return {
        "status": "operational",
        "mode": "lightweight",
        "version": "2.1",
        "environment": "render"
    }


# ================= EXTRACTION ENDPOINT =================

@app.post("/extract")
async def extract_resume(resume: UploadFile = File(...)):
    """Extract text from a resume once to avoid repeated uploads"""
    print(f"\n[EXTRACT] Received: {resume.filename}")
    
    text = extract_content(resume)
    
    if not text or len(text) < 50:
        print(f"[EXTRACT ERROR] Failed or too short: {len(text)} chars")
        return {"status": "error", "text": "", "message": "Extraction failed - check file format"}
        
    print(f"[EXTRACT] Success: {len(text)} chars")
    return {"status": "success", "text": text}


@app.post("/embed-resume")
async def embed_resume(resumeText: str = Form(...)):
    """Extract structured identity and skills from resume text using Groq/Gemini"""
    print(f"\n[EMBED] Processing identity extraction for text len: {len(resumeText)}")
    
    # Check Cache
    cache_key = f"embed:{hashlib.md5(resumeText[:1000].encode()).hexdigest()}"
    if cache_key in AI_RESPONSE_CACHE:
        return AI_RESPONSE_CACHE[cache_key]

    default_result = {
        "name": "",
        "title": "",
        "manifesto": "",
        "skills": [],
        "deployments": [],
        "education": ""
    }

    if groq_client:
        try:
            prompt = f"""
            Analyze the following resume text and extract structured information.
            Return ONLY a JSON object with this exact structure:
            {{
                "name": "Full Name",
                "title": "Current/Most Recent Job Title",
                "manifesto": "A 1-2 sentence professional bio or summary",
                "skills": [
                    {{"skill": "Skill Name", "weight": 1-100 score based on proficiency}}
                ],
                "deployments": [
                    {{"role": "Job Title", "co": "Company Name", "desc": "Brief description of achievements"}}
                ],
                "education": "Degree and University info"
            }}

            Resume Text:
            {resumeText[:3500]}
            """
            
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            res = json.loads(completion.choices[0].message.content)
            
            # Sanitation
            result = {
                "name": res.get("name", ""),
                "title": res.get("title", ""),
                "manifesto": res.get("manifesto", ""),
                "skills": res.get("skills", [])[:15],  # Limit to top 15 skills
                "deployments": res.get("deployments", [])[:5], # Limit to 5 most recent
                "education": res.get("education", "")
            }
            
            AI_RESPONSE_CACHE[cache_key] = result
            return result
        except Exception as e:
            print(f"[Embed Error] LLM fallback: {e}")

    return default_result


# ================= MATCH ENDPOINT =================

@app.post("/match")
async def match_resume(
    resume: UploadFile = File(None),
    resumeText: str = Form(None),
    jobTitle: str = Form(...),
    jobDescription: str = Form(...),
    candidateSkills: str = Form(...),
    profileText: str = Form(""),
    auditSkills: str = Form(None)
):
    print(f"\n{'='*70}")
    print(f"[REQUEST] Job: {jobTitle}")
    if resume:
        print(f"[REQUEST] Resume: {resume.filename}")
    else:
        print(f"[REQUEST] Resume: Text-based ({len(resumeText or '')} chars)")
    print(f"{'='*70}")
    
    # Extract/Assign resume text
    if resumeText:
        resume_text = resumeText
    elif resume:
        resume_text = extract_content(resume)
    else:
        return {
            "fidelityScore": 0,
            "skillAudit": [],
            "matchHighlights": ["No resume provided"],
            "breakdown": {"resume": 0, "profile": 0, "completeness": 0, "skills": 0}
        }
    
    if len(resume_text) < 100:
        print(f"[ERROR] Resume text too short: {len(resume_text)} chars")
        return {
            "fidelityScore": 0,
            "skillAudit": [],
            "matchHighlights": ["Resume content missing or unreadable"],
            "breakdown": {"resume": 0, "profile": 0, "completeness": 0, "skills": 0}
        }
    
    # Step 0: Quick Authenticity Guard
    if not is_authentic_resume(resume_text):
        print("[Guard] Document failed authenticity check. Rejecting.")
        return {
            "matchScore": 0,
            "fidelityScore": 0,
            "skillAudit": [],
            "matchHighlights": ["Document does not appear to be a professional resume (missing key sections or academic/paper markers detected)"],
            "breakdown": {"resume": 0, "profile": 0, "completeness": 0, "skills": 0}
        }
    
    # Prepare inputs
    job_text = f"Role: {jobTitle}\n\n{jobDescription[:2000]}".strip()
    profile_text = profileText[:1500]
    
    print(f"\n[INPUT] Resume: {len(resume_text)} chars")
    print(f"[INPUT] Job: {len(job_text)} chars")
    print(f"[INPUT] Profile: {len(profile_text)} chars")
    
    # Parse skills
    skills = json.loads(candidateSkills)
    audit_skills = json.loads(auditSkills) if auditSkills else []
    
    print(f"[INPUT] Skills: {len(skills)}, Audit: {len(audit_skills)}")
    
    # Tokenize once
    resume_tokens = tokenize(resume_text)
    job_tokens = tokenize(job_text)
    profile_tokens = tokenize(profile_text)
    
    # Compute scores
    print(f"\n[SCORING] Computing scores concurrently...")
    
    # Optimization: Fetch embeddings and compute scores in parallel where possible
    # We'll use a thread pool for the blocking embedding calls
    loop = asyncio.get_event_loop()
    
    async def get_scores():
        # Step 1: Compute resume score and profile score concurrently
        # Note: compute_semantic_score internally handles job_cache now
        resume_task = loop.run_in_executor(None, compute_semantic_score, resume_text, job_text)
        profile_task = loop.run_in_executor(None, compute_semantic_score, profile_text, job_text, 0.03)
        
        res_score, prof_score = await asyncio.gather(resume_task, profile_task)
        return res_score, prof_score

    resume_score, profile_score = await get_scores()
    
    print(f"\n[SCORING] Computing skill overlap...")
    skill_score = compute_skill_match(skills, job_text)
    
    print(f"\n[SCORING] Computing profile quality...")
    quality_score = compute_profile_quality(profile_text, skills)
    
    # Final weighted score: Prioritize Skills (50%) over generic Resume match (30%)
    raw_score = (
        0.30 * resume_score +
        0.50 * skill_score +
        0.20 * (profile_score * 0.7 + quality_score * 0.3)
    )
    
    # SOFT SKILL BOOST (User Request)
    # If the job values soft skills, give a base lift to the raw score
    is_soft_env = detect_soft_skill_environment(job_text)
    if is_soft_env:
        # Increase raw score by 0.12 (12%) to ensure even poor technical matches get visibility
        raw_score += 0.12
        print(f"[Neural] SOFT SKILL BOOST DETECTED: Applying +12% base lift")

    # Linear-Polynomial scaling with penalty for low scores
    # Lower floor for soft-skill environments
    floor = 0.10 if is_soft_env else 0.15
    
    if raw_score < floor:
        final_score_pct = 0
    else:
        # 1.25x power ensures good separation
        distributed_score = raw_score ** 1.25
        final_score_pct = round(distributed_score * 100)
    
    # Final variety offset
    if final_score_pct > 0:
        tie_breaker = (hash(jobTitle) % 3) - 1
        final_score_pct = max(0, min(100, final_score_pct + tie_breaker))
    
    print(f"\n[RESULT] Raw: {raw_score:.3f} → Final: {final_score_pct}%")
    print(f"[BREAKDOWN] Resume: {resume_score:.2%}, Profile: {profile_score:.2%}, Skills: {skill_score:.2%}, Quality: {quality_score:.2%}")
    
    # Skill audit
    skill_audit = [
        {
            "skill": s.upper(),
            "score": compute_skill_mastery(s, resume_tokens, job_tokens, profile_tokens)
        }
        for s in audit_skills[:12]
    ]
    
    # Generate insights
    highlights = generate_highlights(
        resume_tokens,
        job_tokens,
        skills,
        profile_text,
        final_score_pct
    )
    
    print(f"[HIGHLIGHTS] {highlights}\n")
    
    return {
        "fidelityScore": final_score_pct,
        "skillAudit": skill_audit,
        "matchHighlights": highlights,
        "breakdown": {
            "resume": round(resume_score * 100),
            "profile": round(profile_score * 100),
            "completeness": round(quality_score * 100),
            "skills": round(skill_score * 100)
        }
    }


# ================= INSIGHTS =================

@app.post("/insights")
async def insights(candidateName: str = Form(...), jobTitle: str = Form(...), jobDescription: str = Form(""), resumeText: str = Form("")):
    """Generate role-specific insights using Groq"""
    
    # Check Cache
    cache_key = f"insights:{jobTitle}:{hashlib.md5(resumeText[:500].encode()).hexdigest()}"
    if cache_key in AI_RESPONSE_CACHE:
        return AI_RESPONSE_CACHE[cache_key]

    if groq_client and jobDescription and resumeText:
        try:
            prompt = f"""
            Candidate: {candidateName}
            Job Title: {jobTitle}
            Job Description: {jobDescription[:1000]}
            Resume Content: {resumeText[:2000]}
            
            Generate 3 short, punchy, professional insights (one sentence each) about why this candidate is a good match for this specific role. 
            Focus on technical alignment and experience. Return ONLY a JSON list of strings.
            """
            
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            res = json.loads(completion.choices[0].message.content)
            points = res.get("insights") or res.get("points") or list(res.values())[0]
            if isinstance(points, list):
                result = {"points": points[:3]}
                AI_RESPONSE_CACHE[cache_key] = result
                return result
        except Exception as e:
            print(f"[Insights Error] {e}")

    # Fallback
    insights_list = [
        f"Technical competencies align with {jobTitle} requirements",
        "Profile demonstrates relevant domain experience",
        "Skills portfolio matches role expectations"
    ]
    return {"points": insights_list}


@app.post("/tips")
async def tips(jobTitle: str = Form(...), jobDescription: str = Form(...), resumeText: str = Form(...)):
    """Generate personalized interview tips using Groq"""
    
    # Check Cache
    cache_key = f"tips:{jobTitle}:{hashlib.md5(resumeText[:500].encode()).hexdigest()}"
    if cache_key in AI_RESPONSE_CACHE:
        return AI_RESPONSE_CACHE[cache_key]

    if groq_client:
        try:
            prompt = f"""
            Job: {jobTitle}
            JD: {jobDescription[:1000]}
            Resume: {resumeText[:2000]}
            
            Generate personalized interview tips for this candidate.
            Provide:
            1. 3 Strengths based on their resume matching the JD.
            2. 3 Gap areas they should prepare for.
            3. 3 "Power Tips" for the interview.
            
            Return ONLY a JSON object with: {{"strengths": [], "gapAreas": [], "powerTips": []}}
            """
            
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            res = json.loads(completion.choices[0].message.content)
            AI_RESPONSE_CACHE[cache_key] = res
            return res
        except Exception as e:
            print(f"[Tips Error] {e}")

    return None


# ================= SUMMARY =================

@app.post("/summary")
async def summary(jobDescription: str = Form(...)):
    """Extract key requirements and generate a professional summary using Gemini/Groq"""
    
    # Check Cache
    cache_key = f"summary:{hashlib.md5(jobDescription[:500].encode()).hexdigest()}"
    if cache_key in AI_RESPONSE_CACHE:
        return AI_RESPONSE_CACHE[cache_key]

    requirements = ["Technical Skills", "Communication", "Problem Solving"]
    
    if groq_client:
        try:
            prompt = f"Analyze this job description and extract the top 4 technical requirements as short phrases. Also provide a 1-sentence professional summary of the role. Return as JSON: {{\"requirements\": [], \"summary\": \"\"}}\n\nJD: {jobDescription[:2000]}"
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            import json
            res = json.loads(completion.choices[0].message.content)
            result = {
                "requirements": res.get("requirements", requirements)[:4],
                "summary": res.get("summary", "")
            }
            AI_RESPONSE_CACHE[cache_key] = result
            return result
        except Exception as e:
            print(f"[Summary Error] {e}")

    # Fallback to keyword extraction
    tech_keywords = [
        "python", "javascript", "typescript", "java", "react", "angular", "vue",
        "node", "express", "django", "flask", "sql", "mongodb", "postgresql",
        "aws", "azure", "gcp", "docker", "kubernetes", "git", "ci", "cd",
        "api", "rest", "graphql", "frontend", "backend", "fullstack",
        "machine learning", "data", "ai", "cloud", "devops"
    ]
    desc_lower = jobDescription.lower()
    found = [kw.title() for kw in tech_keywords if kw in desc_lower]
    
    return {
        "requirements": found[:4] if found else requirements,
        "summary": "This role focuses on delivering high-quality technical solutions within a dynamic team environment."
    }


# ================= CHAT =================

class ChatRequest(BaseModel):
    jobTitle: str
    jobDescription: str
    question: str
    history: List[dict] = []
    resumeText: str = ""
    matchScore: float = 0.0


@app.post("/chat")
async def chat(req: ChatRequest):
    """Dynamic AI Chat using Groq"""
    
    if groq_client:
        try:
            # Build context-aware system message
            system_msg = f"""
            You are an AI Career Advisor for Asterix-Jobs. 
            Your primary goal is to **only answer exactly what the user asks** while using the provided context (JD, Match Score, Resume) to make your answers accurate and personalized.
            
            **Guidelines:**
            1. **Answer Specifically**: If the user asks a specific question, answer it directly. Do not provide a full audit unless asked.
            2. **Maintain Context**: Remember the previous parts of the conversation and the specific job/candidate details provided.
            3. **Be Concise for Simple Questions**: If the question is simple, give a short answer. If the question is complex, give a detailed answer. (Word limit 250-600 is ONLY for full audits; otherwise, be as brief or as long as necessary).
            4. **Formatting**: Always use clean Markdown (headers, bullet points, bolding) for long answers to ensure readability. Use **double newlines** between sections.
            5. **Tone**: Stay professional, data-driven, and helpful.
            
            Job Context:
            - Title: {req.jobTitle}
            - JD Highlights: {req.jobDescription[:2000]}
            """
            if req.resumeText:
                system_msg += f"""
            Candidate Context:
            - Resume Content: {req.resumeText[:2500]}
            - Match Score: {req.matchScore}%
            """
            
            system_msg += f"""
            Use this context to answer the user's question accurately.
            """
            
            # Convert history to Groq format
            messages = [{"role": "system", "content": system_msg}]
            for msg in req.history[-5:]: # Last 5 messages for context
                # Assuming history is list of {"role": "user/assistant", "content": "..."}
                messages.append(msg)
            
            messages.append({"role": "user", "content": req.question})
            
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.5,
                max_tokens=2000
            )
            
            return {"answer": completion.choices[0].message.content}
        except Exception as e:
            print(f"[Chat Error] {e}")

    # Fallback to simple keyword logic
    q = req.question.lower()
    if "salary" in q or "pay" in q or "compensation" in q:
        answer = "Compensation details are typically discussed during the interview process. Focus on demonstrating your value first."
    elif "skill" in q or "requirement" in q or "qualification" in q:
        answer = f"For the {req.jobTitle} role, review the job description carefully. Key qualifications are typically listed in the requirements section."
    elif "experience" in q or "year" in q:
        answer = "Experience requirements vary by role. Check the job posting for specific details about required years of experience."
    elif "culture" in q or "team" in q or "environment" in q:
        answer = "Company culture and team dynamics are best explored during interviews. Ask about day-to-day responsibilities and team structure."
    else:
        answer = "For specific details about this position, please refer to the complete job description or reach out to the hiring team."
    
    return {"answer": answer}


# ================= EMAIL ENDPOINT =================

class EmailRequest(BaseModel):
    to_email: str
    job_title: str
    company_name: str
    location: str

RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")
print(f"[Startup] RESEND_FROM_EMAIL: {RESEND_FROM_EMAIL}")

@app.post("/send-auto-apply-email")
async def send_auto_apply_email(req: EmailRequest):
    """Notify candidate about automatic application"""
    print(f"\n[EMAIL REQUEST] To: {req.to_email}, Job: {req.job_title}")
    
    if not RESEND_API_KEY:
        print("[Email Error] Cannot send email: RESEND_API_KEY missing")
        return {"status": "error", "message": "Email service not configured"}

    # ... (html_content same)
    html_content = f"""
    <div style="font-family: sans-serif; max-width: 600px; margin: auto; border: 1px solid #eee; padding: 20px;">
        <h2 style="color: #10b981;">Asterix Auto-Pilot Applied For You!</h2>
        <p>Great news! Our AI agent found a high-fidelity match and automatically applied to the following role on your behalf:</p>
        <div style="background: #f9fafb; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <h3 style="margin-top: 0;">{req.job_title}</h3>
            <p style="margin-bottom: 5px;"><strong>Company:</strong> {req.company_name}</p>
            <p style="margin-top: 0;"><strong>Location:</strong> {req.location}</p>
        </div>
        <p>You can track this application in your dashboard.</p>
        <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;" />
        <p style="font-size: 12px; color: #666;">This is an automated notification from your Asterix Neural Protocol.</p>
    </div>
    """

    try:
        print(f"[Email] Attempting to send from {RESEND_FROM_EMAIL} to {req.to_email}...")
        params = {
            "from": RESEND_FROM_EMAIL,
            "to": [req.to_email],
            "subject": f"Applied: {req.job_title} at {req.company_name}",
            "html": html_content,
        }

        email = resend.Emails.send(params)
        print(f"[Email Success] Response ID: {email.get('id')}")
        return {"status": "success", "id": email.get("id")}
    except Exception as e:
        print(f"[Email Error] Exception: {type(e).__name__}: {str(e)}")
        # Check if it's the Resend onboarding email restriction
        if "onboarding@resend.dev" in str(e) or "not verified" in str(e).lower():
            print("[Email Tip] If using onboarding@resend.dev, you can only send to your own authenticated email.")
        return {"status": "error", "message": str(e)}


# ================= PAYMENT ENDPOINTS =================

class OrderRequest(BaseModel):
    amount: float
    customer_id: str
    customer_email: str
    customer_phone: str
    customer_name: str = "Customer"

@app.post("/payments/create-order")
async def create_payment_order(req: OrderRequest):
    """Create a Cashfree order and return session ID"""
    print(f"\n[PAYMENT] Creating order for {req.customer_email}, amount: {req.amount}")
    
    if not CASHFREE_APP_ID or not CASHFREE_SECRET_KEY:
        print("[PAYMENT Error] Cashfree credentials not configured on this server.")
        return {"status": "error", "message": "Payment gateway not configured. Please contact support."}
    
    try:
        customer_details = CustomerDetails(
            customer_id=req.customer_id,
            customer_email=req.customer_email,
            customer_phone=req.customer_phone
        )
        
        # Determine success URL based on origin or default
        # Cashfree PRODUCTION strictly requires HTTPS
        print(f"[PAYMENT] Checking origins for HTTPS: {ALLOWED_ORIGINS}")
        success_url = None
        for origin in ALLOWED_ORIGINS:
            if origin.startswith("https://"):
                success_url = f"{origin}/confirm-payment?order_id={{order_id}}"
                break
        
        # Fallback to main domain if no HTTPS origin found but in Production
        if not success_url:
            success_url = "https://asterix-jobs.in/confirm-payment?order_id={order_id}"
        
        print(f"[PAYMENT] Final return_url: {success_url}")
        
        order_meta = OrderMeta(
            return_url=success_url
        )

        create_order_request = CreateOrderRequest(
            order_amount=req.amount,
            order_currency="INR",
            customer_details=customer_details,
            order_meta=order_meta
        )

        response = Cashfree().PGCreateOrder("2023-08-01", create_order_request)
        
        print(f"[PAYMENT] Raw response type: {type(response)}")
        print(f"[PAYMENT] Raw response.data type: {type(response.data) if response else 'None'}")
        print(f"[PAYMENT] Raw response.data: {response.data if response else 'None'}")
        
        if response and response.data:
            data = response.data
            # SDK may return object or dict — handle both
            if isinstance(data, dict):
                session_id = data.get("payment_session_id")
                order_id = data.get("order_id")
            else:
                session_id = getattr(data, "payment_session_id", None)
                order_id = getattr(data, "order_id", None)
            
            print(f"[PAYMENT Success] Order ID: {order_id}, Session ID: {session_id}")
            
            if not session_id:
                return {"status": "error", "message": "Order created but no payment_session_id returned. Check Cashfree dashboard."}
            
            return {
                "status": "success",
                "payment_session_id": session_id,
                "order_id": order_id
            }
        else:
            print("[PAYMENT Error] Failed to create order")
            return {"status": "error", "message": "Failed to create order"}

    except Exception as e:
        print(f"[PAYMENT Error] {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/payments/status/{order_id}")
async def get_payment_status(order_id: str):
    """Verify payment status with Cashfree"""
    print(f"\n[PAYMENT] Checking status for order: {order_id}")
    
    try:
        response = Cashfree().PGOrderFetchPayments("2023-08-01", order_id)
        
        if response and response.data:
            # Check the status of the first payment (simplified)
            if len(response.data) > 0:
                payment = response.data[0]
                return {
                    "status": "success",
                    "payment_status": payment.payment_status,
                    "order_id": order_id
                }
            
        return {"status": "pending", "order_id": order_id}

    except Exception as e:
        print(f"[PAYMENT Status Error] {str(e)}")
        return {"status": "error", "message": str(e)}


# ================= CONTACT ENDPOINT =================

class ContactRequest(BaseModel):
    name: str
    email: str
    subject: str
    message: str

@app.post("/contact")
async def contact(req: ContactRequest):
    """Handle contact form submissions and send email notification"""
    print(f"\n[CONTACT] From: {req.email}, Subject: {req.subject}")

    if not RESEND_API_KEY:
        print("[Contact Error] Cannot send email: RESEND_API_KEY missing")
        return {"status": "error", "message": "Email service not configured"}

    html_content = f"""
    <div style="font-family: sans-serif; max-width: 600px; margin: auto; border: 1px solid #eee; padding: 20px;">
        <h2 style="color: #000;">New Contact Message from Asterix</h2>
        <div style="background: #f9fafb; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <p><strong>Name:</strong> {req.name}</p>
            <p><strong>Email:</strong> {req.email}</p>
            <p><strong>Subject:</strong> {req.subject}</p>
            <p style="margin-top: 15px;"><strong>Message:</strong></p>
            <p style="white-space: pre-wrap;">{req.message}</p>
        </div>
        <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;" />
        <p style="font-size: 12px; color: #666;">This message was sent via the Asterix Contact Form.</p>
    </div>
    """

    try:
        # Send to jatinthakurr2003@gmail.com
        params = {
            "from": RESEND_FROM_EMAIL,
            "to": ["jatinthakurr2003@gmail.com"],
            "subject": f"Contact: {req.subject} (from {req.name})",
            "html": html_content,
        }

        email = resend.Emails.send(params)
        print(f"[Contact Success] Notification sent to Jatin. ID: {email.get('id')}")
        return {"status": "success", "id": email.get("id")}
    except Exception as e:
        print(f"[Contact Error] {str(e)}")
        return {"status": "error", "message": str(e)}


# ================= JD PARSER =================

class ParseJDRequest(BaseModel):
    text: str

@app.post("/parse-jd")
async def parse_jd(req: ParseJDRequest):
    """Parse raw Job Description text into structured JSON for Admin Auto-Fill"""
    if not req.text or len(req.text) < 10:
        return {"status": "error", "message": "Please paste at least one full sentence of the job description."}
        
    # Check Cache
    cache_key = f"parse_jd:{hashlib.md5(req.text[:2000].encode()).hexdigest()}"
    if cache_key in AI_RESPONSE_CACHE:
        return AI_RESPONSE_CACHE[cache_key]

    default_result = {
        "title": "",
        "companyName": "",
        "city": "",
        "employmentType": "Full-Time",
        "experienceRequired": "",
        "openings": "1",
        "jobSummary": "",
        "responsibilities": [],
        "requiredSkills": [],
        "benefits": []
    }

    if groq_client:
        try:
            prompt = f"""
            Analyze the following raw Job Description text and extract structured information.
            Return ONLY a JSON object with this exact structure (no markdown formatting, just pure JSON):
            {{
                "title": "Job Title (e.g., SDET Intern, Software Engineer)",
                "companyName": "Company Name",
                "city": "Location/City",
                "employmentType": "Full-Time or Part-Time or Contract or Internship",
                "experienceRequired": "Required experience (e.g., 0-1 years)",
                "openings": "Number of openings (default 1 if not specified but keep as string)",
                "jobSummary": "A 2-3 sentence summary of the role/company",
                "responsibilities": ["Responsibility 1", "Responsibility 2"],
                "requiredSkills": ["Core Mandatory Tech Skill 1", "Core Mandatory Tech Skill 2"],
                "preferredSkills": ["Soft/Professional Skill 1", "Soft/Professional Skill 2"],
                "techStack": ["Technology 1", "Technology 2"],
                "benefits": ["Benefit 1", "Benefit 2"]
            }}

            CRITICAL INSTRUCTIONS:
            - All items in "requiredSkills", "preferredSkills", and "techStack" MUST be strict keywords (e.g., "Python", "React", "Docker") NOT full sentences.
            - ABSOLUTELY NO prefixes like "Expertise in...", "Knowledge of...", "Basic understanding of...", "Strong skills in...".
            - Just the raw skill names.
            - "techStack" should contain specific technologies, libraries, or tools.
            - "preferredSkills" should contain soft skills, professional qualities, or secondary requirements.
            - "requiredSkills" should contain the most critical technical must-haves.

            Raw JD Text:
            {req.text[:5000]}
            """
            
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            res = json.loads(completion.choices[0].message.content)
            
            # Sanitation
            result = {
                "title": res.get("title", ""),
                "companyName": res.get("companyName", ""),
                "city": res.get("city", ""),
                "employmentType": res.get("employmentType", "Full-Time"),
                "experienceRequired": str(res.get("experienceRequired", "")),
                "openings": str(res.get("openings", "1")),
                "jobSummary": res.get("jobSummary", ""),
                "responsibilities": res.get("responsibilities", []),
                "requiredSkills": res.get("requiredSkills", []),
                "preferredSkills": res.get("preferredSkills", []),
                "techStack": res.get("techStack", []),
                "benefits": res.get("benefits", [])
            }
            
            AI_RESPONSE_CACHE[cache_key] = result
            return result
        except Exception as e:
            print(f"[Parse JD Error] LLM fallback: {e}")

    return default_result

# ================= CAMPUS CONNECT TEST GENERATION =================

class GenerateTestRequest(BaseModel):
    skills: List[str]

@app.post("/generate-test")
async def generate_test(req: GenerateTestRequest):
    """Generate dynamic test questions based on user skills."""
    if not req.skills:
        return {"status": "error", "message": "At least one skill is required."}

    skills_str = ", ".join(req.skills)
    cache_key = f"gen_test:{hashlib.md5(skills_str.encode()).hexdigest()}"
    
    if cache_key in AI_RESPONSE_CACHE:
        return AI_RESPONSE_CACHE[cache_key]

    default_result = [
        {"question": "What is 2 + 2?", "options": ["3", "4", "5", "6"], "answer": "4", "type": "aptitude", "difficulty": "easy"}
    ]

    if groq_client:
        try:
            prompt = f"""
            You are an expert technical recruiter and assessor.
            Generate a 52-question test for a candidate with the following skills: {skills_str}.
            
            The test must contain exactly:
            - 25 Aptitude / Logical Reasoning multiple-choice questions
            - 25 Technical multiple-choice questions focusing on the provided skills (mix of easy, medium, hard)
            - 2 Coding questions focusing on the provided skills (require written code answers, not multiple choice)

            Return ONLY a JSON object with a single key "questions" containing an array of exactly 52 objects.
            
            Structure for multiple-choice questions (the first 50):
            {{
                "question": "The question text",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "The exact string of the correct option",
                "type": "aptitude or technical",
                "difficulty": "easy, medium, or hard"
            }}

            Structure for coding questions (the final 2):
            {{
                "question": "The problem description and exact requirements...",
                "options": [],
                "answer": "",
                "type": "coding",
                "difficulty": "hard"
            }}
            
            Return the JSON precisely like this:
            {{ "questions": [ ... ] }}
            """
            
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            res = json.loads(completion.choices[0].message.content)
            questions = res.get("questions", default_result)
            
            AI_RESPONSE_CACHE[cache_key] = questions
            return questions
        except Exception as e:
            print(f"[Generate Test Error] LLM fallback: {e}")

    return default_result

# ================= RUN =================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=1,
        log_level="info"
    )