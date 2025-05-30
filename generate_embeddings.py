import os
import pandas as pd
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import time
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """
        Initialize the embedding generator with specified model
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        logger.info(f"Initializing EmbeddingGenerator with model: {model_name}")
        
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load the model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            logger.warning("Empty text list provided. Returning empty array.")
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
        start_time = time.time()
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Embeddings generated successfully in {elapsed_time:.2f} seconds")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

def process_dataframe(df: pd.DataFrame, text_column: str = 'transcription') -> pd.DataFrame:
    """
    Process a dataframe to add embeddings for a specified text column
    """
    logger.info(f"Processing DataFrame with {len(df)} rows")
    
    # Check if the text column exists
    if text_column not in df.columns:
        error_msg = f"Column '{text_column}' not found in DataFrame"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Initialize embedding generator
    generator = EmbeddingGenerator()
    
    # Get non-null texts to embed
    mask = df[text_column].notna()
    texts_to_embed = df.loc[mask, text_column].tolist()
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(texts_to_embed)
    
    # Create a new DataFrame with embeddings
    df_with_embeddings = df.copy()
    
    # Convert embeddings to list (if not already) and then assign
    embedding_list = [emb.tolist() for emb in embeddings] if hasattr(embeddings[0], 'tolist') else list(embeddings)
    
    # Create a Series with the same index as the filtered rows
    embedding_series = pd.Series(embedding_list, index=df.loc[mask].index)
    
    # Add embeddings column if it doesn't exist
    if 'embedding' not in df_with_embeddings.columns:
        df_with_embeddings['embedding'] = None
    
    # Assign the embeddings using the Series
    df_with_embeddings.loc[mask, 'embedding'] = embedding_series
    
    # For null entries, add empty lists
    if (~mask).any():
        logger.warning(f"Found {(~mask).sum()} null entries in '{text_column}' column")
        df_with_embeddings.loc[~mask, 'embedding'] = [[] for _ in range((~mask).sum())]
    
    return df_with_embeddings

def main():
    """Main function to load data, generate embeddings, and save results"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input and output paths
    input_path = os.path.join(script_dir, "data", "parsed_records.parquet")
    output_dir = os.path.join(script_dir, "data", "embeddings")
    output_path = os.path.join(output_dir, "records_with_embeddings.parquet")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {input_path}")
    try:
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded {len(df)} records")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    # Process data to add embeddings
    df_with_embeddings = process_dataframe(df)
    
    # Save results
    logger.info(f"Saving data with embeddings to {output_path}")
    try:
        df_with_embeddings.to_parquet(output_path)
        logger.info("Data successfully saved")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise
    
    return df_with_embeddings

if __name__ == "__main__":
    df = main()
    print(f"\nSuccessfully processed {len(df)} records")
    print(f"Sample embedding shape: {len(df.loc[0, 'embedding']) if len(df) > 0 else 'N/A'}")