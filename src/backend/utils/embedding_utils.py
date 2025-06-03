import os
import logging
from typing import List
import openai
from src.backend.utils.settings import SETTINGS

logger = logging.getLogger(__name__)

if SETTINGS.OPENAI_API_KEY:
    openai.api_key = SETTINGS.OPENAI_API_KEY
else:
    logger.warning("No OpenAI API key found in environment variables")

DEFAULT_MODEL = "text-embedding-3-small"

def generate_embedding(
    text: str, model: str = DEFAULT_MODEL
) -> List[float]:
    """Generate embedding for text using OpenAI API.
    
    Args:
        text: Text to embed
        model: Embedding model to use
    
    Returns:
        List of floats representing the embedding vector
    """
    if not text:
        logger.warning("Empty text provided for embedding generation")
        return []
        
    try:
        response = openai.Embedding.create(
            model=model,
            input=[text]
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []

def batch_generate_embeddings(texts: List[str], model: str = DEFAULT_MODEL) -> List[List[float]]:
    """Generate embeddings for multiple texts.
    
    Args:
        texts: List of texts to embed
        model: Embedding model to use
    
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
        
    # Filter empty strings
    valid_texts = [t for t in texts if t]
    if not valid_texts:
        return [[] for _ in texts]
        
    try:
        response = openai.Embedding.create(
            model=model,
            input=valid_texts
        )
        return [item["embedding"] for item in response["data"]]
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        return [[] for _ in valid_texts]