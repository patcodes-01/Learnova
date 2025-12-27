# nlp/preprocess.py
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(raw_text):
    # Normalize whitespace, remove weird chars
    text = re.sub(r'\r\n', ' ', raw_text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_questions(text):
    # Try splitting by numbered bullets or question marks
    # This is heuristic: improves extraction of single questions
    parts = re.split(r'\d+\.\s+', text)  # split on "1. " "2. " patterns
    if len(parts) <= 1:
        parts = text.split('?')  # fallback: split by question mark
    # strip and drop tiny fragments
    questions = [p.strip() for p in parts if len(p.strip()) > 10]
    return questions

def extract_candidate_nouns(text):
    doc = nlp(text.lower())
    nouns = [token.lemma_ for token in doc if token.pos_ in ('NOUN','PROPN')]
    # filter stopwords/short tokens
    nouns = [t for t in nouns if len(t) > 2]
    return nouns
 
 