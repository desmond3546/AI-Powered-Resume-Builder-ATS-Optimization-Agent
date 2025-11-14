# =====================================
# app.py — Final AI-Powered Resume Builder & ATS Optimizer
# =====================================

import os, io, re, subprocess, tempfile
from typing import Dict
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------
# Load environment & API keys
# -------------------------
load_dotenv()
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Resume Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI-Powered Resume Builder & ATS Optimization Agent")

# -------------------------
# Session state initialization
# -------------------------
for key in ["resume_text", "enhanced_text", "score_history", "feedback_history"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "text" in key else []

# -------------------------
# Templates
# -------------------------
TEMPLATE_MAP = {
    "ModernCV": "templates/moderncv.tex",
    "AutoCV": "templates/autocv.tex"
}

# -------------------------
# Helper functions
# -------------------------
def safe_extract_text_from_pdf(uploaded_file) -> str:
    try:
        pdf = PdfReader(uploaded_file)
        return "\n".join([p.extract_text() or "" for p in pdf.pages])
    except:
        return ""

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

def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def build_latex_resume(template_file: str, fields: Dict) -> bytes:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "resume.tex")
            with open(template_file, "r", encoding="utf-8") as f:
                template = f.read()
            for k, v in fields.items():
                template = template.replace(f"{{{{{k}}}}}", latex_escape(v))
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(template)
            result = subprocess.run(
                ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", tex_path],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                return generate_pdf_from_text(fields.get("summary",""))  # fallback
            pdf_path = tex_path.replace(".tex", ".pdf")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    return f.read()
    except:
        return generate_pdf_from_text(fields.get("summary",""))

# -------------------------
# ATS Score Placeholder
# -------------------------
def get_ats_score_api(resume_text: str) -> dict:
    keywords = ["Python","Java","C++","SQL","AWS","Machine Learning","AI","Data","Project"]
    missing = [kw for kw in keywords if kw.lower() not in resume_text.lower()]
    score = max(20, 100 - len(missing)*10)
    return {"score": score, "missing_keywords": missing}

# -------------------------
# AI Resume Enhancement
# -------------------------
def ai_enhance_resume(text, keywords=[]):
    prompt = f"""
    You are an expert resume writer and ATS specialist.
    Enhance this resume to improve grammar, clarity, and professional tone.
    Ensure the following keywords appear: {', '.join(keywords)}.
    Return ONLY the enhanced resume text.
    """
    client = OpenAI(api_key=OPENAI_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=1200
    )
    return response.choices[0].message.content

# -------------------------
# Step 1 — Input
# -------------------------
st.header("Step 1 — Upload or Enter Resume")
col1, col2 = st.columns([2,1])
with col1:
    method = st.radio("Input method:", ["Upload PDF/DOCX", "Paste / Manual Form"])
    resume_text = ""
    if method == "Upload PDF/DOCX":
        uploaded = st.file_uploader("Upload resume", type=["pdf","docx"])
        if uploaded:
            if uploaded.name.lower().endswith(".pdf"):
                resume_text = safe_extract_text_from_pdf(uploaded)
            else:
                doc = Document(uploaded)
                resume_text = "\n".join(p.text for p in doc.paragraphs)
    else:
        resume_text = st.text_area("Paste or type your resume text here", height=300)

if not resume_text.strip():
    st.info("Provide a resume to continue.")
    st.stop()
st.session_state.resume_text = resume_text

# -------------------------
# Step 2 — Original ATS Score
# -------------------------
st.header("Step 2 — Original ATS Score")
orig_ats = get_ats_score_api(st.session_state.resume_text)
st.metric("Original ATS Score", orig_ats["score"])
st.write(f"Missing keywords: {', '.join(orig_ats['missing_keywords']) if orig_ats['missing_keywords'] else 'None'}")

# -------------------------
# Step 3 — Enhance Resume
# -------------------------
st.header("Step 3 — Enhance Resume with AI")
if st.button("✨ Enhance Resume"):
    with st.spinner("Enhancing resume..."):
        enhanced = ai_enhance_resume(st.session_state.resume_text, orig_ats["missing_keywords"])
        st.session_state.enhanced_text = enhanced
        new_ats = get_ats_score_api(enhanced)
        st.session_state.score_history.append({"orig": orig_ats["score"], "final": new_ats["score"]})
        st.success(f"Enhanced! New ATS Score: {new_ats['score']}")
        st.text_area("Enhanced Resume", value=enhanced, height=300)

# -------------------------
# Step 4 — Template Selection
# -------------------------
st.header("Step 4 — Template Selection")
template_choice = st.selectbox("Choose LaTeX template:", list(TEMPLATE_MAP.keys()))
selected_template_file = TEMPLATE_MAP[template_choice]
st.write(f"Selected template: `{selected_template_file}`")

# -------------------------
# Step 5 — Generate Resume
# -------------------------
st.header("Step 5 — Download Resume")
final_text = st.session_state.enhanced_text or st.session_state.resume_text
fields = {"summary": final_text}

col_pdf, col_docx = st.columns(2)
with col_pdf:
    if st.button("📄 Download PDF"):
        pdf_bytes = build_latex_resume(selected_template_file, fields)
        st.download_button("Download PDF", data=pdf_bytes, file_name="AI_Enhanced_Resume.pdf", mime="application/pdf")

with col_docx:
    if st.button("📥 Download DOCX"):
        docx_bytes = generate_docx_from_text(final_text)
        st.download_button("Download DOCX", data=docx_bytes, file_name="AI_Enhanced_Resume.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# -------------------------
# Step 6 — Score History
# -------------------------
if st.session_state.score_history:
    st.header("ATS Score Improvement History")
    rows = st.session_state.score_history[-12:]
    st.line_chart({"Original": [r["orig"] for r in rows], "Enhanced": [r["final"] for r in rows]})

# -------------------------
# Step 7 — Feedback Chat
# -------------------------
st.sidebar.header("💬 Feedback Chat")
msg = st.sidebar.text_area("Ask for feedback:", height=90)
if st.sidebar.button("Send feedback request"):
    if st.session_state.enhanced_text:
        client = OpenAI(api_key=OPENAI_KEY)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Provide concise actionable feedback on this resume:\n\n{st.session_state.enhanced_text}"}],
            temperature=0.3,
            max_tokens=450
        )
        st.session_state.feedback_history.append({"question": msg, "answer": r.choices[0].message.content})
    else:
        st.session_state.feedback_history.append({"question": msg, "answer": "No enhanced resume to analyze."})

st.sidebar.markdown("### 💬 Feedback Chat History")
for entry in st.session_state.feedback_history[-10:]:
    st.sidebar.markdown(f"**You:** {entry['question']}")
    st.sidebar.markdown(f"**AI:** {entry['answer']}\n---")
