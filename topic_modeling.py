import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
import logging
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Combine embeddings and perform BERTopic modeling"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load embeddings
    sbb_embeddings_path = os.path.join(script_dir, "data", "embeddings", "records_with_embeddings.parquet")
    translation_embeddings_path = os.path.join(script_dir, "data", "embeddings", "translation_embeddings.parquet")
    
    logger.info(f"Loading SBB embeddings from {sbb_embeddings_path}")
    sbb_df = pd.read_parquet(sbb_embeddings_path)
    
    logger.info(f"Loading translation embeddings from {translation_embeddings_path}")
    translation_df = pd.read_parquet(translation_embeddings_path)
    
    # Combine datasets
    combined_docs = list(sbb_df['transcription'].dropna()) + list(translation_df['transcription'].dropna())
    combined_embeddings = np.vstack([
        np.array(list(sbb_df['embedding'].values)),
        np.array(list(translation_df['embedding'].values))
    ])
    
    # Track document sources for later identification
    doc_sources = ['sbb'] * len(sbb_df) + ['translation'] * len(translation_df)
    doc_ids = list(sbb_df['citation']) + list(translation_df['citation'])
    
    # Create BERTopic model
    # Using English as main language but will handle German content through embeddings
    vectorizer = CountVectorizer(stop_words="english")
    topic_model = BERTopic(language="english", vectorizer_model=vectorizer)
    
    # Fit the model using pre-computed embeddings
    topics, probs = topic_model.fit_transform(combined_docs, embeddings=combined_embeddings)
    
    # Create a DataFrame with results
    results_df = pd.DataFrame({
        'source': doc_sources,
        'doc_id': doc_ids,
        'topic': topics,
        'probability': probs
    })
    
    # Save results
    results_path = os.path.join(script_dir, "data", "topic_modeling_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved topic modeling results to {results_path}")
    
    # Visualize topics
    fig = topic_model.visualize_topics()
    plt.savefig(os.path.join(script_dir, "data", "topic_visualization.png"), dpi=300, bbox_inches="tight")
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(script_dir, "data", "topic_info.csv"), index=False)
    
    # Print summary
    logger.info(f"Found {len(topic_model.get_topic_info())} topics")
    logger.info("Top 5 topics:")
    for i, row in topic_model.get_topic_info().head().iterrows():
        if row['Topic'] != -1:  # Skip outlier topic
            words = ', '.join([word for word, _ in topic_model.get_topic(row['Topic'])][:5])
            logger.info(f"Topic {row['Topic']}: {words}")

if __name__ == "__main__":
    main()