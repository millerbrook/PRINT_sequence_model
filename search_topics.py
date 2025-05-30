import os
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import torch
from typing import List, Dict, Any, Tuple, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    def __init__(self, embeddings_path: str, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """
        Initialize the semantic search engine with document embeddings and model
        
        Args:
            embeddings_path: Path to parquet file with document embeddings
            model_name: Name of the SentenceTransformer model to use
        """
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        
        # Load document embeddings
        logger.info(f"Loading document embeddings from {embeddings_path}")
        self.df = self._load_embeddings()
        
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load the model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def _load_embeddings(self) -> pd.DataFrame:
        """Load document embeddings from parquet file"""
        try:
            df = pd.read_parquet(self.embeddings_path)
            logger.info(f"Loaded {len(df)} documents with embeddings")
            
            # Verify embeddings exist and are in the right format
            if 'embedding' not in df.columns:
                raise ValueError("Embedding column not found in DataFrame")
            
            # Check if we have at least one embedding to verify structure
            if len(df) > 0:
                first_embedding = df.iloc[0]['embedding']
                if not isinstance(first_embedding, (list, np.ndarray)):
                    raise ValueError(f"Embeddings not in expected format. Got {type(first_embedding)}")
                
                logger.info(f"Embedding dimension: {len(first_embedding)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.5) -> pd.DataFrame:
        """
        Search for documents similar to the query
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            DataFrame with search results
        """
        logger.info(f"Searching for: '{query}'")
        
        # Generate embedding for query
        query_embedding = self.model.encode(query)
        
        # Calculate similarity between query and all documents
        similarities = []
        
        # Convert embeddings to numpy arrays for vectorized operations
        document_embeddings = np.stack(self.df['embedding'].to_numpy())
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], document_embeddings)[0]
        
        # Add similarities to a copy of the dataframe
        results_df = self.df.copy()
        results_df['similarity'] = similarities
        
        # Filter by threshold and sort
        results_df = results_df[results_df['similarity'] >= threshold]
        results_df = results_df.sort_values('similarity', ascending=False)
        
        # Return top k results
        results = results_df.head(top_k)
        logger.info(f"Found {len(results)} results with similarity >= {threshold}")
        
        return results
    
    def search_topics(self, topics: List[str], top_k: int = 5, threshold: float = 0.5) -> Dict[str, pd.DataFrame]:
        """
        Search for multiple topics and return results for each
        
        Args:
            topics: List of topic phrases to search for
            top_k: Number of top results per topic
            threshold: Minimum similarity score threshold
            
        Returns:
            Dictionary of topic -> search results DataFrame
        """
        results = {}
        for topic in topics:
            results[topic] = self.search(topic, top_k, threshold)
        return results

def format_search_results(results_df: pd.DataFrame, text_column: str = 'transcription', 
                         max_text_length: int = 300) -> List[Dict[str, Any]]:
    """Format search results for display"""
    formatted_results = []
    
    for _, row in results_df.iterrows():
        # Get text and truncate if needed
        text = row.get(text_column, "")
        if isinstance(text, str) and len(text) > max_text_length:
            text = text[:max_text_length] + "..."
            
        result = {
            'citation': row.get('citation', ""),
            'date': row.get('date', ""),
            'sender': row.get('sender', ""),
            'receiver': row.get('receiver', ""),
            'similarity': f"{row.get('similarity', 0):.4f}",
            'text_snippet': text
        }
        formatted_results.append(result)
    
    return formatted_results

def interactive_search(search_engine: SemanticSearchEngine):
    """Run interactive search mode"""
    print("\n===== Interactive Semantic Search =====")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        query = input("\nEnter search query: ")
        if query.lower() in ('exit', 'quit'):
            break
            
        try:
            top_k = int(input("Number of results to show (default: 5): ") or 5)
            threshold = float(input("Minimum similarity threshold (0-1, default: 0.5): ") or 0.5)
            
            results = search_engine.search(query, top_k, threshold)
            formatted = format_search_results(results)
            
            print(f"\nFound {len(formatted)} results for '{query}':")
            for i, result in enumerate(formatted, 1):
                print(f"\n--- Result {i} (Similarity: {result['similarity']}) ---")
                print(f"Citation: {result['citation']}")
                print(f"Date: {result['date']}")
                print(f"Sender: {result['sender']} -> Receiver: {result['receiver']}")
                print(f"Text snippet: {result['text_snippet']}")
                
        except Exception as e:
            print(f"Error during search: {str(e)}")

def batch_search(search_engine: SemanticSearchEngine, topics: List[str], 
                top_k: int = 5, threshold: float = 0.5, output_file: Optional[str] = None):
    """Run batch search for specified topics"""
    results_dict = search_engine.search_topics(topics, top_k, threshold)
    
    # Process and display results
    all_formatted_results = {}
    
    print("\n===== Batch Search Results =====")
    for topic, results in results_dict.items():
        formatted = format_search_results(results)
        all_formatted_results[topic] = formatted
        
        print(f"\n\nResults for topic: '{topic}'")
        print(f"Found {len(formatted)} results with similarity >= {threshold}")
        
        for i, result in enumerate(formatted, 1):
            print(f"\n--- Result {i} (Similarity: {result['similarity']}) ---")
            print(f"Citation: {result['citation']}")
            print(f"Text snippet: {result['text_snippet']}")
    
    # Save results to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_formatted_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_file}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Semantic search for documents')
    
    # Main parameters
    parser.add_argument('--embeddings', type=str, 
                        default='data/embeddings/records_with_embeddings.parquet',
                        help='Path to document embeddings')
    parser.add_argument('--model', type=str, 
                        default='paraphrase-multilingual-mpnet-base-v2',
                        help='SentenceTransformer model name')
    
    # Search modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--interactive', action='store_true',
                          help='Run in interactive mode')
    mode_group.add_argument('--topics', type=str,
                          help='Comma-separated list of topics to search for')
    
    # Search parameters
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top results to return per topic')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum similarity threshold (0-1)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for batch search results (JSON)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(script_dir, args.embeddings)
    
    # Initialize search engine
    search_engine = SemanticSearchEngine(embeddings_path, args.model)
    
    # Run in selected mode
    if args.interactive:
        interactive_search(search_engine)
    else:
        # Parse topics
        topics = [t.strip() for t in args.topics.split(',')]
        
        # Set output path if specified
        output_path = None
        if args.output:
            output_path = os.path.join(script_dir, args.output)
            
        # Run batch search
        batch_search(search_engine, topics, args.top_k, args.threshold, output_path)

if __name__ == "__main__":
    main()