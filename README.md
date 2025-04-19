
# 📄 Earnings Call Analyzer – Streamlit Web App

A **Streamlit-based web application** that enables users to upload **earnings call transcripts in PDF format**, extract clean textual data (ignoring tables), and generate **detailed AI-powered reports** using **Google Gemini**. Users can also ask **custom questions** and download the extracted sentences as a CSV file for further analysis.

---

## 🚀 Features

- 📤 Upload and process PDF files
- 🧾 Extract only textual content (tables are ignored)
- ✂️ Clean and tokenize text into individual sentences
- 🤖 Generate AI-powered reports summarizing the content
- ❓ Ask custom questions based on the extracted text
- 📥 Download the extracted sentences as a CSV file

---

## 🛠️ Technologies Used

- **Python**
- **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/en/latest/)** – for accurate PDF text extraction
- **[nltk](https://www.nltk.org/)** – for natural language text tokenization
- **[Google Generative AI (Gemini)](https://ai.google.dev/)** – for summarization and Q&A
- **[Pandas](https://pandas.pydata.org/)** – for handling structured data and CSV export
- **[Streamlit](https://streamlit.io/)** – for building the web interface

---

## ⚙️ Installation

### 1. Prerequisites

- Python 3.8 or newer installed on your machine

### 2. Install Dependencies

Run the following command in your terminal:

```bash
pip install pymupdf pandas nltk google-generativeai streamlit
```

### 3. Configure Google Gemini API

Replace the placeholder API key in `main.py` with your actual Gemini API key:

```python
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

> 🔐 Ensure you keep your API key secure and do not expose it in public repositories.

---

## ▶️ Running the Application

In the terminal, navigate to the project directory and run:

```bash
streamlit run main.py
```

The app will launch in your default web browser.

---

## 📁 File Structure

```
📂 Project Directory
│-- main.py                # Main Streamlit app
│-- uploads/               # Stores uploaded PDF files
│-- extracted_text.csv     # CSV output of extracted sentences
```

---

## 💡 Example Output – Sample Q&A

**Q:** What are the key risks mentioned?

**A:** The report highlights potential supply chain disruptions and increasing costs as major risks.

---

## 📌 Notes

- The app focuses on **earnings call transcripts** but can be adapted to other types of business or analytical documents.
- Tables and figures in PDFs are skipped to maintain clean textual data for NLP processing.
- The CSV export makes it easier to conduct downstream analysis or share insights.

---

## 📬 Feedback & Contributions

Feel free to open an issue or pull request if you have suggestions or improvements!

---
