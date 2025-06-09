import os
import zipfile
import re
import pandas as pd
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import torch
import time
from typing import List, Dict, Optional
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranslationExtractor:
    def __init__(self, zip_path: str, extract_dir: str = "temp_extracts"):
        """
        Initialize extractor for translations from ZIP file
        
        Args:
            zip_path: Path to the ZIP file
            extract_dir: Directory to extract files to (temporary)
        """
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        
    def extract_files(self) -> List[str]:
        """Extract files from ZIP and return paths to English translation files"""
        logger.info(f"Extracting files from {self.zip_path}")
        
        # Create extraction directory if it doesn't exist
        os.makedirs(self.extract_dir, exist_ok=True)
        
        # Extract all files
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_dir)
        
        # Find all English translation files
        en_files = []
        for root, _, files in os.walk(self.extract_dir):
            for file in files:
                if 'EN' in file and file.endswith('_translate'):
                    en_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(en_files)} English translation files")
        return en_files
    
    def extract_translations(self, files: List[str]) -> Dict[str, str]:
        """Extract translation text from files"""
        translations = {}
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract text after "Translation:"
                match = re.search(r'Translation:(.*?)(?:\n\w+:|$)', content, re.DOTALL | re.IGNORECASE)
                if match:
                    translation_text = match.group(1).strip()
                    file_id = os.path.basename(file_path)
                    translations[file_id] = translation_text
                else:
                    logger.warning(f"No translation found in {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Extracted translations from {len(translations)} files")
        return translations
    
    def cleanup(self):
        """Remove temporary extraction directory"""
        if os.path.exists(self.extract_dir):
            shutil.rmtree(self.extract_dir)
            logger.info(f"Removed temporary directory {self.extract_dir}")

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """Initialize the embedding generator with specified model"""
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
        """Generate embeddings for a list of texts"""
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

def create_dataframe_from_translations(translations: Dict[str, str], embeddings: np.ndarray) -> pd.DataFrame:
    """Create DataFrame with translations and embeddings"""
    # Create records
    records = []
    file_ids = list(translations.keys())
    
    for i, file_id in enumerate(file_ids):
        # Parse file ID to extract metadata
        parts = file_id.split('_')
        citation = parts[0] if len(parts) > 0 else ""
        
        record = {
            'file_id': file_id,
            'citation': citation,
            'transcription': translations[file_id],
            'embedding': embeddings[i].tolist() if i < len(embeddings) else []
        }
        records.append(record)
    
    return pd.DataFrame(records)

def main():
    """Main function to extract translations and generate embeddings"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input and output paths
    zip_path = os.path.join(script_dir, "data", "generated_docs_6_5_25_v2.zip")
    output_dir = os.path.join(script_dir, "data", "embeddings")
    output_path = os.path.join(output_dir, "translation_embeddings.parquet")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract translations
    extractor = TranslationExtractor(zip_path)
    try:
        en_files = extractor.extract_files()
        translations = extractor.extract_translations(en_files)
    finally:
        extractor.cleanup()  # Ensure temp files are cleaned up
    
    # Generate embeddings
    if translations:
        generator = EmbeddingGenerator()
        texts = list(translations.values())
        embeddings = generator.generate_embeddings(texts)
        
        # Create DataFrame
        df = create_dataframe_from_translations(translations, embeddings)
        
        # Save to Parquet
        df.to_parquet(output_path)
        logger.info(f"Saved {len(df)} translation embeddings to {output_path}")
        
        # Display sample
        if len(df) > 0:
            logger.info(f"Sample embedding dimension: {len(df.iloc[0]['embedding'])}")
            logger.info(f"Sample translation (truncated): {df.iloc[0]['transcription'][:100]}...")
    else:
        logger.warning("No translations found to process")

if __name__ == "__main__":
    main()