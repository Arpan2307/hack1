import os
import fitz  # PyMuPDF
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify

# Configure Google Gemini API key
genai.configure(api_key="AIzaSyDVeEWQBc5aCqPSdXMbC1rmygI33CLuqXw")  # Replace with your Gemini API key

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Process PDF
    raw_text = extract_text_from_pdf(file_path)
    clean_text = preprocess_text(raw_text)
    sentences = text_to_sentences(clean_text)

    # Save as CSV
    csv_path = os.path.join(app.config["UPLOAD_FOLDER"], "extracted_text.csv")
    pd.DataFrame(sentences, columns=["Sentence"]).to_csv(csv_path, index=True)

    # Generate Report
    report = generate_report(clean_text)

    return jsonify({"report": report, "csv_path": csv_path})

@app.route("/ask_questions", methods=["POST"])
def ask_questions():
    data = request.json
    report = data.get("report", "")
    questions = data.get("questions", [])

    if not report or not questions:
        return jsonify({"error": "Missing report or questions"}), 400

    if len(questions) > 3:
        return jsonify({"error": "You can ask a maximum of 3 questions"}), 400

    response = analyze_questions(report, questions)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
