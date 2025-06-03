from typing import List, Dict, Any
import json
import logging
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import numpy as np
import jax
import jax.numpy as jnp  # JAX NumPy replacement
import Stemmer  # PyStemmer
from src.backend.utils.settings import SETTINGS

logger = logging.getLogger(__name__)

class SearchMetadata(BaseModel):
    category: str
    keywords: List[str]
    related_topics: List[str]


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: SearchMetadata


class BM25S:
    """BM25S - Extension of BM25 with improved saturation parameters."""
    
    def __init__(self, tokenized_docs, k1=1.5, b=0.75, delta=0.5):
        """Initialize BM25S with tokenized documents and parameters.
        
        Args:
            tokenized_docs: List of tokenized documents (each doc is a list of terms)
            k1: Term saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
            delta: Smoothing parameter for improved term saturation (default: 0.5)
        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.corpus_size = len(tokenized_docs)
        self.avgdl = sum(len(doc) for doc in tokenized_docs) / self.corpus_size
        self.tokenized_docs = tokenized_docs
        
        # Create document frequency dictionary
        self.doc_freqs = {}
        for doc in tokenized_docs:
            doc_terms = set(doc)
            for term in doc_terms:
                if term not in self.doc_freqs:
                    self.doc_freqs[term] = 0
                self.doc_freqs[term] += 1
        
        # Create term frequency matrix
        self.term_freqs = []
        for doc in tokenized_docs:
            term_freq = {}
            for term in doc:
                if term not in term_freq:
                    term_freq[term] = 0
                term_freq[term] += 1
            self.term_freqs.append(term_freq)
    
    def get_scores(self, query):
        """Calculate BM25S scores for a query across all documents.
        
        Args:
            query: List of terms in the query
            
        Returns:
            List of BM25S scores for each document
        """
        scores = np.zeros(self.corpus_size)
        
        for term in set(query):
            if term not in self.doc_freqs:
                continue
                
            # Calculate IDF
            idf = np.log(1 + (self.corpus_size - self.doc_freqs[term] + 0.5) / 
                          (self.doc_freqs[term] + 0.5))
            
            # Calculate score contribution for each document
            for i, doc in enumerate(self.tokenized_docs):
                doc_len = len(doc)
                if term not in self.term_freqs[i]:
                    continue
                
                # Term frequency in document
                freq = self.term_freqs[i][term]
                
                # BM25S score with delta parameter for better saturation
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + self.delta
                scores[i] += idf * numerator / denominator
        
        return scores


class HybridRetriever:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.client = chromadb.PersistentClient(
            path=self.cfg.hybrid_retriever.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=SETTINGS.OPENAI_API_KEY,
            model_name=self.cfg.llm.embedding_model
        )
        self.collection = self.client.get_collection(
            name=self.cfg.hybrid_retriever.collection,
            embedding_function=self.embedding_function
        )
        
        # Initialize stemmer from PyStemmer
        self.stemmer = Stemmer.Stemmer('english')
        
    def _sanitize_text(self, text: str) -> str:
        """Clean text and handle control characters."""
        if not text or not isinstance(text, str):
            return text
        
        # Replace control characters
        import re
        text = text.replace('\x03', ' ')
        text = text.replace('\x0093', '"')
        text = text.replace('\x0094', '"')
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
        return text
        
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalization of scores using JAX for better performance."""
        if not scores:
            return scores
        
        # Convert to JAX array for optimized operations
        jax_scores = jnp.array(scores)
        min_score = jnp.min(jax_scores)
        max_score = jnp.max(jax_scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        # Normalize using JAX operations
        normalized = (jax_scores - min_score) / (max_score - min_score)
        return normalized.tolist()

    def _tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize and stem text for improved matching."""
        # Basic tokenization (replace with more sophisticated if needed)
        tokens = text.lower().split()
        
        # Apply stemming
        stemmed_tokens = self.stemmer.stemWords(tokens)
        
        # Filter out stop words and very short tokens
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                      'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of'}
        filtered_tokens = [token for token in stemmed_tokens if token not in stop_words and len(token) > 1]
        
        return filtered_tokens

    def _get_keyword_scores(self, query: str, documents: List[str]) -> List[float]:
        """Get BM25S scores with stemming for improved keyword matching."""
        # Clean and sanitize all texts
        query = self._sanitize_text(query)
        documents = [self._sanitize_text(doc) for doc in documents]
        
        # Tokenize and stem documents
        tokenized_docs = [self._tokenize_and_stem(doc) for doc in documents]
        
        # Tokenize and stem query
        tokenized_query = self._tokenize_and_stem(query)
        
        # Create BM25S index with custom parameters
        bm25s = BM25S(tokenized_docs, k1=1.5, b=0.75, delta=0.5)
        
        # Get BM25S scores
        keyword_scores = bm25s.get_scores(tokenized_query)
        
        # Normalize scores
        return self._normalize_scores(keyword_scores.tolist())

    async def search(
        self, query: str, filter_conditions: Dict = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching with enhanced algorithms
        
        Args:
            query: Search query string
            filter_conditions: Optional filters for metadata fields
        """
        logger.debug("Searching with query: %s, but NOT adding documents", query)
        # Sanitize query
        clean_query = self._sanitize_text(query)
        
        # Get semantic search results with scores
        results = self.collection.query(
            query_texts=[clean_query],
            n_results=self.cfg.hybrid_retriever.top_k,
            where=filter_conditions,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Accessing the inner lists of results
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Convert distances to similarity scores (1 - distance) using JAX
        distances = jnp.array(results['distances'][0])
        semantic_scores = self._normalize_scores((1 - distances).tolist())
        
        # Get enhanced keyword search scores with BM25S and stemming
        keyword_scores = self._get_keyword_scores(clean_query, documents)
        
        # Combine scores with configurable weights using JAX for performance
        semantic_weight = self.cfg.hybrid_retriever.semantic_weight
        keyword_weight = self.cfg.hybrid_retriever.keyword_weight
        
        combined_scores = jnp.add(
            jnp.multiply(jnp.array(semantic_scores), semantic_weight),
            jnp.multiply(jnp.array(keyword_scores), keyword_weight)
        ).tolist()
        
        # Sort results by combined score
        search_results = []
        for doc, meta, score in zip(documents, metadatas, combined_scores):
            try:
                keywords = json.loads(meta.get('keywords', '[]'))
                topics = json.loads(meta.get('related_topics', '[]'))
            except (json.JSONDecodeError, TypeError):
                keywords = []
                topics = []
                
            metadata_object = SearchMetadata(
                category=meta.get('category', ''),
                keywords=keywords,
                related_topics=topics
            )

            search_results.append(
                SearchResult(
                    content=doc,
                    score=score,
                    metadata=metadata_object
                )
            )
            
        return sorted(search_results, key=lambda x: x.score, reverse=True)
    
