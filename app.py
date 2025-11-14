# =====================================
# app.py — AI Resume Builder with ATS Optimization
# =====================================
import os
import io
import re
import subprocess
import tempfile
from typing import Tuple, List, Dict

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# AI libs
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------
# Load environment
# -------------------------
load_dotenv()

# -------------------------
# Load API keys
# -------------------------
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="AI Resume Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Resume Builder & ATS Optimization Agent")
st.markdown("Upload or paste your resume, enhance with AI (OpenAI / Gemini) and download ATS-optimized DOCX/PDF.")

st.sidebar.markdown("### API Keys")
st.sidebar.write("OpenAI:", "✅" if OPENAI_KEY else "❌")
st.sidebar.write("Gemini:", "✅" if GEMINI_KEY else "❌")
# -------------------------
# AI engine choice (sidebar)
# -------------------------
st.sidebar.header("AI Engine Selection")
use_gemini = st.sidebar.checkbox("Use Gemini (Google) instead of OpenAI", value=False)

# -------------------------
# Session state
# -------------------------
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "enhanced_text" not in st.session_state:
    st.session_state.enhanced_text = ""
if "score_history" not in st.session_state:
    st.session_state.score_history = []
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

# -------------------------
# AI engine choice
# -------------------------
use_gemini = st.checkbox("Use Gemini (Google) instead of OpenAI", value=False)

# -------------------------
# Helper functions
# -------------------------
def safe_extract_text_from_pdf(uploaded_file) -> str:
    try:
        pdf = PdfReader(uploaded_file)
        return "\n".join([p.extract_text() or "" for p in pdf.pages])
    except:
        return ""

def get_ats_score_local(text: str) -> int:
    keywords = ["Python","Java","C++","SQL","AWS","Machine Learning","AI","Data","Project"]
    count = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text, re.I))
    return min(100, count * 10)

def cached_ai_score(text: str) -> int:
    key = f"ai_score_{hash(text)}"
    if key in st.session_state:
        return st.session_state[key]
    score = ai_score_api(text)
    st.session_state[key] = score
    return score

def ai_score_api(text: str) -> int:
    prompt = f"Rate this resume on a scale of 0-100 for ATS compatibility:\n{text}"
    try:
        if use_gemini and GEMINI_KEY:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            resp = model.generate_content(prompt)
            score = int(re.findall(r"\d+", resp.text)[0])
        elif OPENAI_KEY:
            client = OpenAI(api_key=OPENAI_KEY)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=50
            )
            score = int(re.findall(r"\d+", r.choices[0].message.content)[0])
        else:
            score = get_ats_score_local(text)
    except:
        score = get_ats_score_local(text)
    return min(max(score,0),100)

def cached_ai_enhance(text: str, prompt: str, keywords: List[str]=[]) -> str:
    key = f"ai_enhance_{hash(text)}"
    if key in st.session_state:
        return st.session_state[key]
    enhanced = ai_enhance_api(text, prompt, keywords)
    st.session_state[key] = enhanced
    return enhanced

def ai_enhance_api(text: str, prompt: str, keywords: List[str]=[]) -> str:
    full_prompt = f"{prompt}\n\n{text}\n\nEnsure the following keywords are present: {', '.join(keywords)}"
    try:
        if use_gemini and GEMINI_KEY:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            resp = model.generate_content(full_prompt)
            return resp.text
        elif OPENAI_KEY:
            client = OpenAI(api_key=OPENAI_KEY)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":full_prompt}],
                temperature=0.2,
                max_tokens=1200
            )
            return r.choices[0].message.content
        else:
            return text + "\n" + ", ".join(keywords)
    except Exception as e:
        st.warning(f"AI enhancement failed, using fallback. Error: {e}")
        return text + "\n" + ", ".join(keywords)

def re_inject_keywords(original: str, enhanced: str) -> Tuple[str, List[str]]:
    keywords = ["Python","Java","C++","SQL","AWS","Machine Learning","AI","Data","Project"]
    missing = []
    for kw in keywords:
        if re.search(rf"\b{kw}\b", original, re.I) and not re.search(rf"\b{kw}\b", enhanced, re.I):
            enhanced += f"\n{kw}"
            missing.append(kw)
    return enhanced, missing

def generate_docx_from_text(text: str) -> bytes:
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

def generate_pdf_from_text(text: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flowables = [Paragraph(line, styles["Normal"]) for line in text.split("\n")]
    doc.build(flowables)
    buf.seek(0)
    return buf.getvalue()

# -------------------------
# Step 1 — Upload / Manual
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
                except:
                    extracted = ""
            if extracted.strip():
                st.session_state.resume_text = extracted
    else:
        name = st.text_input("Full name", "")
        email = st.text_input("Email", "")
        phone = st.text_input("Phone", "")
        linkedin = st.text_input("LinkedIn / profile", "")
        github = st.text_input("GitHub / profile", "")
        edu = st.text_area("Education (brief)", height=80)
        skills = st.text_area("Skills (comma or bullet list)", height=80)
        work = st.text_area("Work experience (paste bullets)", height=120)
        projects = st.text_area("Projects / achievements", height=100)
        summary = st.text_area("Summary / Objective (optional)", height=80)
        if st.button("✅ Merge manual entries into Resume Text"):
            parts = []
            for v,label in [(name,""),(email,""),(phone,""),(linkedin,""),(github,""),
                            (summary,"\nSummary:\n"),(edu,"\nEducation:\n"),
                            (skills,"\nSkills:\n"),(work,"\nExperience:\n"),(projects,"\nProjects:\n")]:
                if v:
                    parts.append(label+v)
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

# Editor
st.subheader("Resume Text Editor")
st.session_state.resume_text = st.text_area("Resume Text (you can edit)", value=st.session_state.resume_text, height=260, key="resume_editor")

# -------------------------
# Step 2 — AI ATS Score
# -------------------------
st.header("Step 2 — Compute AI-based ATS Score")
if st.button("Compute AI ATS Score"):
    with st.spinner("Calculating AI score..."):
        st.session_state.ai_score = cached_ai_score(st.session_state.resume_text)

st.metric("Local ATS Score", f"{get_ats_score_local(st.session_state.resume_text)} / 100")
if st.session_state.get("ai_score") is not None:
    st.metric("AI-based ATS Score", f"{st.session_state.ai_score} / 100")

# -------------------------
# Step 3 — Enhance Resume
# -------------------------
st.header("Step 3 — Enhance Resume with AI")
enhance_prompt = """You are an expert technical resume writer and ATS specialist.
Enhance the resume text to improve clarity, grammar, formatting and to preserve technical keywords.
Keep the resume concise and professional. Return ONLY the resume content (no commentary)."""

if st.button("✨ Enhance Resume"):
    with st.spinner("Enhancing resume..."):
        enhanced = cached_ai_enhance(st.session_state.resume_text, enhance_prompt)
        enhanced, missing = re_inject_keywords(st.session_state.resume_text, enhanced)
        st.session_state.enhanced_text = enhanced
        st.session_state.score_history.append({
            "orig": cached_ai_score(st.session_state.resume_text),
            "final": cached_ai_score(enhanced)
        })
        st.success(f"Enhanced! New ATS Score: {st.session_state.score_history[-1]['final']}")
        if missing:
            st.info(f"Re-injected keywords: {', '.join(missing)}")

if st.session_state.enhanced_text:
    st.subheader("Enhanced Resume (Preview)")
    st.text_area("Enhanced Resume", value=st.session_state.enhanced_text, height=320)

# -------------------------
# Step 4 & 5 — Templates + Download
# -------------------------
TEMPLATE_MAP = {"ModernCV": "templates/moderncv.tex", "AutoCV": "templates/autocv.tex"}
st.header("Step 4 — Template Selection")
template_choice = st.selectbox("Choose a LaTeX template:", list(TEMPLATE_MAP.keys()))
selected_template_file = TEMPLATE_MAP[template_choice]
st.write(f"Selected template file: `{selected_template_file}`")

st.header("Step 5 — Generate Resume (PDF / DOCX)")
final_text = st.session_state.enhanced_text or st.session_state.resume_text
col_pdf, col_docx = st.columns(2)

# LaTeX PDF generator with fallback
with col_pdf:
    if st.button("📄 Generate PDF"):
        try:
            # Prepare fields for template replacement
            fields = {
                "resume_text": final_text
            }
            # Attempt LaTeX compilation
            pdf_bytes = build_latex_resume(selected_template_file, fields)
        except Exception as e:
            st.warning(f"LaTeX PDF generation failed, using fallback. Error: {e}")
            pdf_bytes = generate_pdf_from_text(final_text)
        st.download_button(
            "📥 Download PDF",
            pdf_bytes,
            file_name="AI_Enhanced_Resume.pdf",
            mime="application/pdf"
        )

# DOCX download
with col_docx:
    if st.button("📥 Download DOCX"):
        docx_bytes = generate_docx_from_text(final_text)
        st.download_button(
            "📥 Download DOCX",
            docx_bytes,
            file_name="AI_Enhanced_Resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

# -------------------------
# Score Comparison
# -------------------------
st.header("Comparison & Score Tracker")
c1, c2 = st.columns(2)
with c1: st.text_area("Original Resume", value=st.session_state.resume_text, height=300)
with c2: st.text_area("Enhanced Resume", value=st.session_state.enhanced_text or st.session_state.resume_text, height=300)
if st.session_state.score_history:
    st.subheader("Score Improvement History")
    rows = st.session_state.score_history[-12:]
    st.line_chart({"original": [r["orig"] for r in rows], "final": [r["final"] for r in rows]})

# -------------------------
# Sidebar Feedback Chat
# -------------------------
st.sidebar.header("💬 Quick Feedback Chat")
msg = st.sidebar.text_area("Ask for feedback:", height=90)
if st.sidebar.button("Send feedback request"):
    with st.spinner("Getting feedback..."):
        try:
            full_resume = st.session_state.enhanced_text or st.session_state.resume_text
            if use_gemini and GEMINI_KEY:
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                resp = model.generate_content(f"Provide actionable feedback on this resume:\n\n{full_resume}")
                st.session_state.feedback_history.append({"question": msg, "answer": resp.text})
            elif OPENAI_KEY:
                client = OpenAI(api_key=OPENAI_KEY)
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":f"Provide actionable feedback on this resume:\n\n{full_resume}"}],
                    temperature=0.3,
                    max_tokens=450
                )
                st.session_state.feedback_history.append({"question": msg, "answer": r.choices[0].message.content})
            else:
                st.session_state.feedback_history.append({"question": msg, "answer": "No AI key configured."})
        except Exception as e:
            st.session_state.feedback_history.append({"question": msg, "answer": f"Feedback failed: {e}"})

st.sidebar.markdown("### 💬 Feedback Chat")
for entry in st.session_state.feedback_history[-10:]:
    st.sidebar.markdown(f"**You:** {entry['question']}")
    st.sidebar.markdown(f"**AI:** {entry['answer']}\n---")

