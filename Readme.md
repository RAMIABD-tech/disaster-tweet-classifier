# 🚨 Real-Time Emergency Tweet Classifier (NLP)

An end-to-end Natural Language Processing (NLP) web application that classifies social media text into **Real Emergencies** vs. **Non-Disasters / Metaphorical Slang**.

## 📌 Problem Statement
During critical emergencies, first responders and relief agencies monitor social media to extract real-time crisis data. However, simple keyword filtering fails when words like *"fire"*, *"flood"*, or *"ablaze"* are used metaphorically (e.g., *"This new album is fire 🔥"* vs. *"The building is on fire 🚒"*). 

This project trains a Machine Learning pipeline to distinguish contextual intent in short tweets.

---

## 📊 Results & Performance
* **Dataset:** 10,000+ hand-labeled tweets from Kaggle.
* **Baseline F1 Score (Raw Text):** `0.7448`
* **Cleaned F1 Score (Regex Preprocessed):** `0.7462`
* **Key Finding:** Cleaning URLs, HTML entities, and special characters reduced noise and improved F1 classification score.

---

## 🛠️ Tech Stack & Methods
* **Language:** Python
* **Data Processing & EDA:** Pandas, NumPy, Regex (`re`)
* **Vectorization:** TF-IDF (`TfidfVectorizer`)
* **Machine Learning:** Logistic Regression (`scikit-learn`)
* **Deployment:** Streamlit, Joblib
