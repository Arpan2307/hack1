import os
import fitz  # PyMuPDF
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
import streamlit as st

# Configure Google Gemini API key
genai.configure(api_key="AIzaSyAc2tMGUFkMK3G68KaHb3Vi-KMmYlRv7Oo")  # Replace with your Gemini API key

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def is_table(block):
    """Check if a block of text looks like a table."""
    lines = block["lines"]
    if len(lines) > 2:
        spans_per_line = [len(line["spans"]) for line in lines]
        avg_spans = sum(spans_per_line) / len(spans_per_line)
        if avg_spans > 3 and max(spans_per_line) - min(spans_per_line) <= 2:
            return True
    return False

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file while ignoring tables."""
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0 and not is_table(block):
                for line in block["lines"]:
                    for span in line["spans"]:
                        text += span["text"]
                    text += '\n'
                text += '\n'
        text += '\f'
    return text

def preprocess_text(text):
    """Clean text by removing URLs, emails, and extra whitespace."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_sentences(text):
    """Convert text to sentences while merging very short ones."""
    sentences = sent_tokenize(text)
    if not sentences:
        return []
    merged_sentences = []
    prev = sentences[0]
    for sen in sentences:
        if len(sen) < 6:
            prev += " " + sen
        else:
            merged_sentences.append(prev)
            prev = sen
    merged_sentences.append(prev)
    return merged_sentences

def generate_report(text):
    """Generate a detailed report using Gemini AI."""
    prompt = f"""
    Analyze the following earnings call transcript and provide a detailed, layman's report.
    
    - Summarize the key points.
    - Identify financial highlights (growth, revenue, profits).
    - Highlight potential risks and red flags in simple terms.
    
    Make sure the language is easy to understand for someone without financial expertise.
    
    Transcript:
    {text}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def analyze_questions(report, questions):
    """Pass user questions along with the report to Gemini AI for better insights."""
    question_text = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(questions)])
    prompt = f"""
    Based on the following report, answer the user's questions in detail:
    Report:
    {report}
    
    User Questions:
    {question_text}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# Streamlit App
st.title("📄 PDF Analysis and Reporting")

uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")
if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ File uploaded: {uploaded_file.name}")

    # Process PDF
    raw_text = extract_text_from_pdf(file_path)
    clean_text = preprocess_text(raw_text)
    sentences = text_to_sentences(clean_text)

    # Save as CSV
    csv_path = os.path.join(UPLOAD_FOLDER, "extracted_text.csv")
    pd.DataFrame(sentences, columns=["Sentence"]).to_csv(csv_path, index=True)

    # Generate Report
    st.write("### 📝 Generated Report")
    report = generate_report(clean_text)
    st.markdown(report.replace("-", "- 🟢"))

    # Add question-answer feature to the Streamlit app
    st.write("### ❓ Ask Questions")
    questions = [st.text_input(f"Question {i+1}") for i in range(3)]
    if st.button("Submit Questions"):
        if not questions or all(q.strip() == "" for q in questions):
            st.error("Please enter at least one question.")
        else:
            response = analyze_questions(report, [q for q in questions if q.strip()])
            st.write("### 🤖 AI Responses")
            st.markdown(response.replace("-", "- 🟢"))

    # Download CSV
    st.download_button(
        label="📥 Download Extracted Sentences as CSV",
        data=open(csv_path, "rb").read(),
        file_name="extracted_text.csv",
        mime="text/csv"
    )