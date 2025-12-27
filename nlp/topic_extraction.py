# nlp/topic_extraction.py
from rake_nltk import Rake
from collections import Counter
from .preprocess import split_into_questions, clean_text

rake = Rake()

def extract_topics_rake(full_text, top_k=50):
    """
    Returns ranked keyword phrases from the whole PYQ text.
    """
    text = clean_text(full_text)
    rake.extract_keywords_from_text(text)
    phrases = rake.get_ranked_phrases()
    # basic normalization
    phrases = [p.lower().strip() for p in phrases if len(p.strip()) > 1]
    return phrases[:top_k]

# If you want to extract topics per question:
def extract_topics_per_question(full_text):
    text = clean_text(full_text)
    questions = split_into_questions(text)
    per_q_topics = []
    for q in questions:
        rake.extract_keywords_from_text(q)
        phrases = rake.get_ranked_phrases()
        per_q_topics.append([p.lower().strip() for p in phrases if len(p.strip())>1])
    return per_q_topics

