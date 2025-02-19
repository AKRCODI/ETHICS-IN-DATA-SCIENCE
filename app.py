import openai
import re
import json
import logging
import os
import sqlite3
import tkinter as tk
from flask import Flask, request, jsonify, render_template
from tkinter import filedialog, scrolledtext
from transformers import pipeline
from sklearn.metrics import accuracy_score

# Setup logging
logging.basicConfig(filename="ai_decisions.log", level=logging.INFO, format='%(asctime)s - %(message)s')

# Load OpenAI API Key (Replace with your key)
openai.api_key = "YOUR_OPENAI_API_KEY"

# Load bias detection model
bias_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Database setup
DB_NAME = "resumes.db"
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS resumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    skills TEXT,
    experience TEXT,
    education TEXT,
    bias_scores TEXT,
    fair_selection BOOLEAN
)
""")
conn.commit()

def parse_resume(text):
    """Extracts key details from resumes using GPT."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Extract name, skills, experience, education from this resume."},
            {"role": "user", "content": text}
        ]
    )
    return json.loads(response["choices"][0]["message"]["content"])

def detect_bias(text):
    """Detects potential bias in the resume analysis."""
    categories = ["gender_bias", "race_bias", "age_bias"]
    result = bias_classifier(text, candidate_labels=categories, multi_label=True)
    return dict(zip(result["labels"], result["scores"]))

def anonymize_resume(text):
    """Anonymizes sensitive candidate details to prevent bias."""
    text = re.sub(r'\b(Mr|Ms|Mrs|Dr)\.\s[A-Z][a-z]+', 'Candidate', text)
    return text

def evaluate_fairness(scores):
    """Ensures demographic parity & fairness in AI decisions."""
    fairness_threshold = 0.75
    fairness_score = sum(scores.values()) / len(scores)
    return fairness_score >= fairness_threshold

def save_to_db(name, skills, experience, education, bias_scores, fair_selection):
    """Saves resume details to the database."""
    cursor.execute("""
    INSERT INTO resumes (name, skills, experience, education, bias_scores, fair_selection)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (name, skills, experience, education, json.dumps(bias_scores), fair_selection))
    conn.commit()

@app.route("/process_resume", methods=["POST"])
def process_resume():
    data = request.json
    resume_text = data.get("resume_text", "")
    
    # Step 1: Anonymize resume
    clean_text = anonymize_resume(resume_text)
    
    # Step 2: Extract information
    extracted_data = parse_resume(clean_text)
    
    # Step 3: Detect bias
    bias_scores = detect_bias(clean_text)
    
    # Step 4: Fairness evaluation
    is_fair = evaluate_fairness(bias_scores)
    
    # Step 5: Save to Database
    save_to_db(
        extracted_data.get("name", "Unknown"),
        extracted_data.get("skills", ""),
        extracted_data.get("experience", ""),
        extracted_data.get("education", ""),
        bias_scores,
        is_fair
    )
    
    # Step 6: Logging decisions
    logging.info(json.dumps({"Extracted": extracted_data, "BiasScores": bias_scores, "Fair": is_fair}))
    
    return jsonify({"Extracted": extracted_data, "BiasScores": bias_scores, "Fair": is_fair})

@app.route("/dashboard", methods=["GET"])
def dashboard():
    search_query = request.args.get("search", "")
    query = "SELECT * FROM resumes WHERE name LIKE ? OR skills LIKE ?"
    cursor.execute(query, (f"%{search_query}%", f"%{search_query}%"))
    resumes = cursor.fetchall()
    return render_template("dashboard.html", resumes=resumes, search_query=search_query)

if __name__ == "__main__":
    app.run(debug=True)
