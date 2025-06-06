import pandas as pd
import spacy
from langdetect import detect, LangDetectException
import os

def detect_language(text):
    """Detect the language of the given text."""
    try:
        lang = detect(text)
        # langdetect returns 'en' for English, 'de' for German, etc.
        return lang
    except LangDetectException:
        return "unknown"

def extract_entities_from_text(text, nlp):
    """Extract named entities from text using spaCy."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data", "parsed_records.csv")
    output_path = os.path.join(script_dir, "data", "parsed_records_with_ner_multilang.csv")

    # Load the parsed records
    print(f"Loading records from {input_path}...")
    df = pd.read_csv(input_path)

    # Load spaCy models
    print("Loading spaCy models...")
    nlp_en = spacy.load("en_core_web_sm")
    nlp_de = spacy.load("de_core_news_sm")

    # Detect language and perform NER
    languages = []
    named_entities = []

    print("Detecting language and extracting named entities...")
    for text in df["transcription"].fillna(""):
        lang = detect_language(text)
        languages.append(lang)
        if lang == "en":
            entities = extract_entities_from_text(text, nlp_en)
        elif lang == "de":
            entities = extract_entities_from_text(text, nlp_de)
        else:
            entities = []
        named_entities.append(entities)

    df["language"] = languages
    df["named_entities"] = named_entities

    # Save the results
    df.to_csv(output_path, index=False)
    print(f"NER results saved to {output_path}")

    # Show a sample
    print("\nSample with language and named entities:")
    print(df[["transcription", "language", "named_entities"]].head())

if __name__ == "__main__":
    main()