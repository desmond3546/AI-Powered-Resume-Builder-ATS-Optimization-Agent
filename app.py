# app.py — Finished AI Resume Builder (full)
from dotenv import load_dotenv
import os
load_dotenv()  # must run before os.getenv()
# Load API keys from env
OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # type: Optional[str]
GEMINI_KEY = os.getenv("GEMINI_API_KEY")  # type: Optional[str]

import io
import random
import requests
import subprocess
import tempfile
import streamlit as st
from typing import Tuple, List, Dict
from PyPDF2 import PdfReader
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# AI libs (user must install google-generativeai and openai-python client)
import google.generativeai as genai
from openai import OpenAI

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI Resume Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Resume Builder & ATS Optimization Agent")
st.markdown("Upload or paste your resume, enhance with AI (OpenAI / Gemini) and download ATS-optimized DOCX/PDF.")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

st.sidebar.markdown("### API Keys")
st.sidebar.write("OpenAI:", "✅" if OPENAI_KEY else "❌")
st.sidebar.write("Gemini:", "✅" if GEMINI_KEY else "❌")

# -------------------------
# Utilities & Helpers
# -------------------------
KEYWORDS = [
    "Python", "Machine Learning", "AI", "Data", "Developer", "Analysis", "Engineer",
    "TensorFlow", "Pandas", "NumPy", "Scikit-learn", "Streamlit", "SQL"
]

# weights for scoring: adjust to taste
WEIGHTS = {
    "python": 3, "machine learning": 4, "ai": 3, "data": 2,
    "developer": 2, "analysis": 2, "engineer": 3,
    "tensorflow": 2, "pandas": 2, "numpy": 2, "scikit-learn": 2, "streamlit": 2, "sql": 2
}

def safe_extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    try:
        reader = PdfReader(uploaded_file)
        for p in reader.pages:
            text += (p.extract_text() or "") + "\n"
    except Exception:
        return ""
    return text

def get_ats_score_local(text: str) -> float:
    t = (text or "").lower()
    score = 0
    for kw, w in WEIGHTS.items():
        hits = min(t.count(kw.lower()), 3)
        score += hits * w
    max_score = sum(WEIGHTS.values()) * 3
    if max_score == 0:
        return 0.0
    return round((score / max_score) * 100, 2)

def get_ats_score_api(text: str) -> float:
    """
    Try to call a free/mock ATS API; if unavailable, fallback to local scoring.
    Replace the mock URL with a real API endpoint when available.
    """
    try:
        resp = requests.post("https://mock-ats-api.fly.dev/score", json={"resume_text": text}, timeout=4)
        if resp.status_code == 200:
            return float(resp.json().get("score", 0))
    except Exception:
        pass
    return get_ats_score_local(text)
def ats_score_ai(resume_text: str, use_gemini_flag: bool) -> float:
    prompt = f"Evaluate the following resume for ATS keywords and give a score out of 100:\n\n{resume_text}"
    try:
        resp_text = call_ai("", prompt, use_gemini_flag)  # text can be empty for scoring
        return float(resp_text.strip().split()[0])
    except Exception:
        return get_ats_score_local(resume_text)



def re_inject_keywords(orig: str, enhanced: str, keywords=KEYWORDS) -> Tuple[str, List[str]]:
    missing = [k for k in keywords if k.lower() in (orig or "").lower() and k.lower() not in (enhanced or "").lower()]
    if missing:
        enhanced = (enhanced or "").strip() + "\n\nTechnical Keywords: " + ", ".join(missing)
    return enhanced, missing

def generate_docx_from_text(text: str) -> io.BytesIO:
    doc = Document()
    doc.add_heading("Resume (AI-Enhanced)", level=0)
    for line in (text or "").split("\n"):
        if line.strip():
            doc.add_paragraph(line.strip())
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

def generate_pdf_from_text(text: str) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []
    for para in (text or "").split("\n"):
        if para.strip():
            elems.append(Paragraph(para.strip().replace("&", "&amp;"), styles["Normal"]))
            elems.append(Spacer(1, 6))
    doc.build(elems)
    buffer.seek(0)
    return buffer

# -------------------------
# AI Enhancement wrappers
# -------------------------
def enhance_with_openai(text: str, prompt: str) -> str:
    if not OPENAI_KEY:
        raise RuntimeError("OpenAI API key not configured.")
    client = OpenAI(api_key=OPENAI_KEY)
    # uses chat.completions - matches user's installed client pattern
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt + "\n\n" + (text or "")}],
        temperature=0.2,
        max_tokens=1200
    )
    return resp.choices[0].message.content

def enhance_with_gemini(text: str, prompt: str) -> str:
    if not GEMINI_KEY:
        raise RuntimeError("Gemini API key not configured.")
    # Prefer fast flash, then pro. Use model names present in your account.
    for m in ("models/gemini-2.5-flash", "models/gemini-2.5-pro", "models/gemini-flash-latest"):
        try:
            model = genai.GenerativeModel(m)
            resp = model.generate_content(prompt + "\n\n" + (text or ""))
            return resp.text
        except Exception:
            continue
    raise RuntimeError("No working Gemini model available for generate_content.")
def call_ai(text: str, prompt: str, use_gemini_flag: bool) -> str:
    """
    Calls either OpenAI or Gemini based on use_gemini_flag.
    Returns the AI-generated text.
    """
    if use_gemini_flag:
        if not GEMINI_KEY:
            raise RuntimeError("Gemini API key not configured.")
        # try multiple Gemini models
        for m in ("models/gemini-2.5-flash", "models/gemini-2.5-pro", "models/gemini-flash-latest"):
            try:
                model = genai.GenerativeModel(m)
                resp = model.generate_content(prompt + "\n\n" + (text or ""))
                return resp.text
            except Exception:
                continue
        raise RuntimeError("No working Gemini model available.")
    else:
        if not OPENAI_KEY:
            raise RuntimeError("OpenAI API key not configured.")
        client = OpenAI(api_key=OPENAI_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt + "\n\n" + (text or "")}],
            temperature=0.2,
            max_tokens=1200
        )
        return resp.choices[0].message.content

    

# -------------------------
# LaTeX template helpers
# -------------------------
TEMPLATE_MAP = {
    "ModernCV (LaTeX)": "moderncv.tex",
    "AutoCV (LaTeX)": "autocv.tex"
}

def extract_sections(enhanced_text: str) -> Dict[str, str]:
    """
    Try to extract common blocks from the enhanced resume text for template replacement.
    This is simple heuristic extraction expecting headings like 'Summary', 'Skills', etc.
    """
    out = {k: "" for k in ("NAME","EMAIL","PHONE","SUMMARY","SKILLS","EXPERIENCE","PROJECTS","EDUCATION","LINKEDIN","GITHUB")}
    txt = (enhanced_text or "").strip()
    if not txt:
        return out
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    out["NAME"] = lines[0] if lines else ""
    # scan for email/phone/link
    for l in lines[:8]:
        if "@" in l and not out["EMAIL"]:
            out["EMAIL"] = l
        if any(ch.isdigit() for ch in l) and len([c for c in l if c.isdigit()]) >= 6 and not out["PHONE"]:
            out["PHONE"] = l
        if "linkedin" in l.lower() and not out["LINKEDIN"]:
            out["LINKEDIN"] = l
        if "github" in l.lower() and not out["GITHUB"]:
            out["GITHUB"] = l
    lower = txt.lower()
    def grab(section_name):
        s = section_name.lower()
        if s in lower:
            start = lower.index(s)
            # find next double newline or end
            end = lower.find("\n\n", start)
            if end == -1:
                return txt[start:].strip()
            return txt[start:end].strip()
        return ""
    out["SUMMARY"] = grab("summary")
    out["SKILLS"] = grab("skills")
    out["EXPERIENCE"] = grab("experience")
    out["PROJECTS"] = grab("projects")
    out["EDUCATION"] = grab("education")
    # Clean section headers if present (remove the header word)
    for k in ("SUMMARY","SKILLS","EXPERIENCE","PROJECTS","EDUCATION"):
        out[k] = out[k].split(":",1)[-1].strip() if out[k] else ""
    return out

def fill_latex_template(template_path: str, data: Dict[str,str]) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        tex = f.read()
    # Replace placeholders like <NAME>, <SUMMARY>, etc.
    for k, v in data.items():
        tex = tex.replace(f"<{k}>", v or "")
    return tex

def build_latex_resume(template_name: str, data: Dict[str,str]) -> str:
    """
    Fill LaTeX template from templates/ and compile to PDF. Returns path to PDF or None.
    """
    template_path = os.path.join("templates", template_name)
    if not os.path.exists(template_path):
        st.error(f"Template file not found: {template_path}")
        return None

    tex_content = fill_latex_template(template_path, data)
    tmp_dir = tempfile.mkdtemp()
    tex_file = os.path.join(tmp_dir, "resume.tex")
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(tex_content)

    try:
        # try to compile (pdflatex must be present). run twice for references (safe)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "resume.tex"], cwd=tmp_dir,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "resume.tex"], cwd=tmp_dir,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        # LaTeX failed — show error in UI and fallback later
        st.error("LaTeX compilation failed. Falling back to simple PDF.")
        return None

    pdf_path = os.path.join(tmp_dir, "resume.pdf")
    if os.path.exists(pdf_path):
        return pdf_path
    return None

# -------------------------
# Session state init
# -------------------------
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "enhanced_text" not in st.session_state:
    st.session_state.enhanced_text = ""
if "score_history" not in st.session_state:
    st.session_state.score_history = []

# -------------------------
# Choose AI engine
# -------------------------
use_gemini = st.checkbox("Use Gemini (Google) instead of OpenAI", value=False)


# -------------------------
# UI: Step 1 — Input
# -------------------------
st.header("Step 1 — Upload or Enter Resume")
col1, col2 = st.columns([2,1])
with col1:
    method = st.radio("Input method:", ["Upload PDF/DOCX", "Paste / Manual Form"])
    if method == "Upload PDF/DOCX":
        uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf","docx"])
        if uploaded:
            if uploaded.name.lower().endswith(".pdf"):
                extracted = safe_extract_text_from_pdf(uploaded)
            else:
                try:
                    doc = Document(uploaded)
                    extracted = "\n".join(p.text for p in doc.paragraphs)
                except Exception:
                    extracted = ""
            if extracted.strip():
                st.session_state.resume_text = extracted
    else:
        # manual form fields (helps LaTeX auto-fill)
        name = st.text_input("Full name", value="")
        email = st.text_input("Email", value="")
        phone = st.text_input("Phone", value="")
        linkedin = st.text_input("LinkedIn / profile", value="")
        github = st.text_input("GitHub / profile", value="")
        edu = st.text_area("Education (brief)", height=80)
        skills = st.text_area("Skills (comma or bullet list)", height=80)
        work = st.text_area("Work experience (paste bullets)", height=120)
        projects = st.text_area("Projects / achievements", height=100)
        summary = st.text_area("Summary / Objective (optional)", height=80)
        if st.button("✅ Merge manual entries into Resume Text"):
            # build simple text structure to feed AI or templates
            parts = []
            if name: parts.append(name)
            if email: parts.append(email)
            if phone: parts.append(phone)
            if linkedin: parts.append(linkedin)
            if github: parts.append(github)
            if summary:
                parts.append("\nSummary:\n" + summary)
            if edu:
                parts.append("\nEducation:\n" + edu)
            if skills:
                parts.append("\nSkills:\n" + skills)
            if work:
                parts.append("\nExperience:\n" + work)
            if projects:
                parts.append("\nProjects:\n" + projects)
            st.session_state.resume_text = "\n".join(parts)
            st.success("Manual inputs copied into resume text editor below.")

with col2:
    st.subheader("Original ATS Score (live)")
    live_score = get_ats_score_local(st.session_state.resume_text)
    st.metric("Original ATS Score", f"{live_score} / 100")
    if st.session_state.score_history:
        st.line_chart([r["final"] for r in st.session_state.score_history])

if not st.session_state.resume_text.strip():
    st.info("Upload a resume or paste / create one using the manual form to continue.")
    st.stop()

# show extracted text
st.subheader("Extracted / Entered Resume Text")
st.text_area("Resume Text (you can edit)", value=st.session_state.resume_text, height=260, key="resume_editor")
# keep edits
if st.session_state.get("resume_editor") is not None:
    st.session_state.resume_text = st.session_state.resume_editor

# Compute all ATS scores
local_score = get_ats_score_local(st.session_state.resume_text)

try:
    ai_score = ats_score_ai(st.session_state.resume_text, use_gemini)
except Exception:
    ai_score = None

try:
    api_score = get_ats_score_api(st.session_state.resume_text)
except Exception:
    api_score = None

# Display scores
st.subheader("ATS Scores Comparison")
st.metric("Local ATS Score", f"{local_score} / 100")
if ai_score is not None:
    st.metric("AI-based ATS Score", f"{ai_score} / 100", delta=(ai_score - local_score))
else:
    st.info("AI-based score not available (requires API key).")

if api_score is not None:
    st.metric("External API ATS Score", f"{api_score} / 100", delta=(api_score - local_score))

# -------------------------
# UI: Step 2 & 3 — ATS score & AI Enhancement
# -------------------------
st.header("Step 2 — ATS Scoring  & Step 3 — AI Enhancement")
st.write("Keywords used for ATS scoring:", ", ".join(KEYWORDS))

prompt = """You are an expert technical resume writer and ATS specialist.
Enhance the resume text to improve clarity, grammar, formatting and to preserve technical keywords.
Keep the resume concise and professional. Return ONLY the resume content (no commentary)."""

if st.button("✨ Enhance Resume with AI"):
    with st.spinner("Enhancing resume..."):
        try:
            enhanced = call_ai(st.session_state.resume_text, prompt, use_gemini)
        except Exception as e:
            st.error(f"AI enhancement failed: {e}")
            enhanced = ""

        if enhanced:
            enhanced, missing = re_inject_keywords(st.session_state.resume_text, enhanced)
            st.session_state.enhanced_text = enhanced
            # update scores and history
            orig_score = ats_score_ai(st.session_state.resume_text, use_gemini)
            new_score = ats_score_ai(st.session_state.enhanced_text, use_gemini)
            st.session_state.score_history.append({"orig": orig_score, "final": new_score})
            st.success(f"Enhanced! New ATS Score: {new_score}")
            # show missing reinjected keywords info
            if missing:
                st.info(f"The following keywords were re-inserted to preserve ATS matches: {', '.join(missing)}")

if st.session_state.enhanced_text:
    st.subheader("Enhanced Resume (Preview)")
    st.text_area("Enhanced Resume", value=st.session_state.enhanced_text, height=320)

# -------------------------
# Step 4 — Template selection (ModernCV / AutoCV only)
# -------------------------
st.header("Step 4 — Template Selection")
template_choice = st.selectbox("Choose a LaTeX template:", list(TEMPLATE_MAP.keys()))
selected_template_file = TEMPLATE_MAP[template_choice]
st.write(f"Selected template file: `{selected_template_file}`")

# -------------------------
# Step 5 — Generate & Download
# -------------------------
st.header("Step 5 — Generate Resume (PDF / DOCX)")

final_text = st.session_state.enhanced_text or st.session_state.resume_text
orig_live = ats_score_ai(st.session_state.resume_text, use_gemini)
final_score = ats_score_ai(final_text, use_gemini)
st.metric("Original ATS Score", f"{orig_live} / 100", delta=None)
st.metric("Final ATS Score", f"{final_score} / 100", delta=(final_score - orig_live))


col_download_pdf, col_download_docx = st.columns(2)

with col_download_pdf:
    if st.button("📄 Generate PDF from LaTeX"):
        with st.spinner("Filling template and compiling LaTeX..."):
            # extract structured fields for template placeholders
            fields = extract_sections(final_text)
            # fill template and compile
            template_path = os.path.join("templates", selected_template_file)
            pdf_path = build_latex_resume(selected_template_file, fields)
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.success("PDF generated using LaTeX!")
                st.download_button("📥 Download LaTeX PDF", data=pdf_bytes, file_name="AI_Enhanced_Resume.pdf", mime="application/pdf")
            else:
                # fallback simple PDF from plain text
                pdf_buf = generate_pdf_from_text(final_text)
                st.warning("LaTeX compile failed or not available — offering fallback PDF.")
                st.download_button("📥 Download Fallback PDF", data=pdf_buf, file_name="AI_Enhanced_Resume.pdf", mime="application/pdf")

with col_download_docx:
    if st.button("📥 Download DOCX (plain)"):
        docx_bytes = generate_docx_from_text(final_text)
        st.download_button("Download DOCX", data=docx_bytes, file_name="AI_Enhanced_Resume.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# -------------------------
# Comparison & History
# -------------------------
st.header("Comparison & Score Tracker")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Original Resume")
    st.text_area("Original", value=st.session_state.resume_text, height=300)
with c2:
    st.subheader("Enhanced Resume")
    st.text_area("Enhanced", value=st.session_state.enhanced_text or st.session_state.resume_text, height=300)

if st.session_state.score_history:
    st.subheader("Score Improvement History")
    rows = st.session_state.score_history[-12:]
    origs = [r["orig"] for r in rows]
    finals = [r["final"] for r in rows]
    st.line_chart({"original": origs, "final": finals})

# -------------------------
# Sidebar Feedback Chat (light)
# -------------------------
st.sidebar.header("💬 Quick Feedback Chat")
if st.session_state.resume_text:
    msg = st.sidebar.text_area("Ask for feedback:", height=90)
    if st.sidebar.button("Send feedback request"):
        with st.sidebar.spinner("Getting feedback..."):
            try:
                if use_gemini and GEMINI_KEY:
                    model = genai.GenerativeModel("models/gemini-2.5-flash")
                    resp = model.generate_content(f"Provide concise actionable feedback on this resume:\n\n{st.session_state.enhanced_text or st.session_state.resume_text}")
                    st.sidebar.write(resp.text)
                elif OPENAI_KEY:
                    client = OpenAI(api_key=OPENAI_KEY)
                    r = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":f"Provide concise actionable feedback on this resume:\n\n{st.session_state.enhanced_text or st.session_state.resume_text}"}],
                        temperature=0.3,
                        max_tokens=450
                    )
                    st.sidebar.write(r.choices[0].message.content)
                else:
                    st.sidebar.info("No AI key configured.")
            except Exception as e:
                st.sidebar.error(f"Feedback call failed: {e}")


