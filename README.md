**Overview**

main.py file is a Streamlit-based web application that allows users to upload PDF files, extract text while ignoring tables, preprocess the text, and generate detailed reports using Google's Gemini AI. Additionally, users can ask specific questions related to the extracted content.


**Features**

Upload and process PDF files

Extract text while ignoring tables

Clean and tokenize text into sentences

Generate AI-powered reports on earnings call transcripts

Allow users to ask custom questions about the extracted content

Download extracted sentences as a CSV file


**Technologies Used**

Python

PyMuPDF (fitz) for PDF text extraction

Natural Language Toolkit (nltk) for text tokenization

Google Gemini AI for content generation

Pandas for CSV data handling

Streamlit for web application interface


**Installation**

Prerequisites

Ensure you have Python installed (recommended: Python 3.8 or newer).

Install Required Packages

Run the following command to install the necessary dependencies:

pip install pymupdf pandas nltk google-generativeai streamlit

Configure Google Gemini API Key

Replace the placeholder API key in the script with your own Gemini API key:

genai.configure(api_key="YOUR_GEMINI_API_KEY")


To run the website, following command is to be given on the terminal:
```streamlit run main.py```

[Users can also download the extracted sentences as a CSV file]

**File Structure**

📂 Project Directory
│-- main.py  # Main Streamlit application file
│-- uploads/  # Directory for storing uploaded files
│-- extracted_text.csv  # CSV file with extracted sentences

**Example Output - Sample Q&A**

Q: What are the key risks mentioned?
A: The report highlights potential supply chain disruptions and increasing costs...
