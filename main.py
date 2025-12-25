# main.py
from ocr.ocr import extract_text_from_pdf
from nlp.preprocess import clean_text
from nlp.topic_extraction import extract_topics_rake
from analysis.frequency_analysis import compute_topic_frequency
from summary.ai_summary import summarize_topics

def run_pipeline(pdf_path, use_snider=False, snider_key=None, use_gemini=False, gemini_key=None):
    print("=== OCR extraction ===")
    text = extract_text_from_pdf(pdf_path, use_snider=use_snider, snider_key=snider_key)
    print("Length of extracted text:", len(text))

    print("\n=== Topic extraction (RAKE) ===")
    phrases = extract_topics_rake(text, top_k=200)
    print("Example extracted phrases:", phrases[:10])

    print("\n=== Frequency analysis ===")
    ranked = compute_topic_frequency(phrases, top_n=10)
    print("Top topics (topic, count, weight%):")
    for topic, cnt, wt in ranked:
        print(topic, cnt, wt)

    print("\n=== Summarization ===")
    top_topic_names = [t for t,_,_ in ranked]
    summaries = summarize_topics(top_topic_names, use_gemini=use_gemini, gemini_key=gemini_key)
    for t in top_topic_names:
        print("\n---", t, "---")
        print(summaries[t])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PYQ PDF")
    args = parser.parse_args()
    run_pipeline(args.pdf)

