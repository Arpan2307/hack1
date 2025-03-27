import os
import fitz  # PyMuPDF
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from flask import Flask, request, render_template, send_file
import google.generativeai as genai

nltk.download('punkt')

# Configure Google Gemini API
genai.configure(api_key="AIzaSyDVeEWQBc5aCqPSdXMbC1rmygI33CLuqXw")  # Replace with your API Key

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def is_table(block):
    """Determine if a block of text is a table"""
    lines = block["lines"]
    if len(lines) > 2:
        spans_per_line = [len(line["spans"]) for line in lines]
        avg_spans = sum(spans_per_line) / len(spans_per_line)
        return avg_spans > 3 and max(spans_per_line) - min(spans_per_line) <= 2
    return False


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
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
    return text


def preprocess_text(text):
    """Clean text by removing URLs, emails, and extra whitespace"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d{4}.\d{2}.\d{2} \d{2}:\d{2}:\d{2} [+-]\d{2}\'\d{2}\'', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_to_sentences(text):
    """Split text into meaningful sentences"""
    sentences = sent_tokenize(text)
    if not sentences:
        return []
    
    merged_sentences = []
    prev = sentences[0]
    
    for sen in sentences:
        if len(sen) < 6:
            prev += sen
        else:
            merged_sentences.append(prev)
            prev = sen

    merged_sentences.append(prev)
    return merged_sentences


def analyze_text(text_data):
    """Generate insights using Gemini AI"""
    prompt = (
        "Analyze the following transcript and provide key insights:\n"
        "1. Summarize key points.\n"
        "2. Identify financial highlights (growth, revenue, margins, etc.).\n"
        f"Transcript:\n{text_data}"
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "pdf_file" not in request.files:
            return "No file uploaded", 400
        
        pdf_file = request.files["pdf_file"]
        if pdf_file.filename == "":
            return "No selected file", 400
        
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
        pdf_file.save(file_path)

        # Process PDF
        raw_text = extract_text_from_pdf(file_path)
        cleaned_text = preprocess_text(raw_text)
        sentences = text_to_sentences(cleaned_text)

        # Save sentences to CSV
        csv_output_path = os.path.join(OUTPUT_FOLDER, "output.csv")
        pd.DataFrame(sentences, columns=['Sentence']).to_csv(csv_output_path, index=False)

        # Analyze the extracted text
        analysis_result = analyze_text(cleaned_text)

        return render_template("index.html", analysis=analysis_result, csv_file=csv_output_path)

    return render_template("index.html")


@app.route("/download")
def download_csv():
    return send_file(os.path.join(OUTPUT_FOLDER, "output.csv"), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
