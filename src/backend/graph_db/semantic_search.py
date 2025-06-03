import logging
from typing import List, Dict, Any, Optional
from src.backend.graph_db.neo4j_manager import db_manager
from src.backend.utils.embedding_utils import generate_embedding, batch_generate_embeddings

logger = logging.getLogger(__name__)

# --- Vector Index Management ---
def create_vector_indexes(dimensions: int = 1536):
    """Create vector indexes for all entity types.
    
    Args:
        dimensions: Dimension of the embedding vectors
    """
    try:
        # Create index for candidates
        candidate_query = f"""
        CREATE VECTOR INDEX candidate_embeddings IF NOT EXISTS
        FOR (c:Candidate) ON (c.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        db_manager.run_query(candidate_query)
        
        # Create index for constituencies
        constituency_query = f"""
        CREATE VECTOR INDEX constituency_embeddings IF NOT EXISTS
        FOR (c:Constituency) ON (c.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        db_manager.run_query(constituency_query)
        
        # Create index for parties
        party_query = f"""
        CREATE VECTOR INDEX party_embeddings IF NOT EXISTS
        FOR (p:Party) ON (p.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        db_manager.run_query(party_query)
        
        logger.info("Successfully created all vector indexes")
        return True
    except Exception as e:
        logger.error(f"Error creating vector indexes: {e}")
        return False

# --- Embedding Generation and Update ---
def update_candidate_embeddings():
    """Update embeddings for all candidate nodes."""
    try:
        # Get candidates with bio text
        query = """
        MATCH (c:Candidate)
        WHERE c.bio IS NOT NULL AND c.bio <> ""
        RETURN id(c) as id, c.name as name, c.bio as bio
        """
        candidates = db_manager.run_query(query)
        
        # Process in batches
        batch_size = 20
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            texts = [f"{c['name']}: {c['bio']}" for c in batch]
            embeddings = batch_generate_embeddings(texts)
            
            # Update candidates with embeddings
            for j, candidate in enumerate(batch):
                update_query = """
                MATCH (c:Candidate)
                WHERE id(c) = $id
                SET c.embedding = $embedding
                """
                db_manager.run_query(update_query, id=candidate['id'], embedding=embeddings[j])
        
        logger.info(f"Updated embeddings for {len(candidates)} candidates")
        return True
    except Exception as e:
        logger.error(f"Error updating candidate embeddings: {e}")
        return False

# --- Semantic Search Functions ---
def semantic_search_candidates(
    query: str,
    limit: int = 5,
    min_score: float = 0.7, 
    constituency: Optional[str] = None, 
    party: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search for candidates using semantic similarity.
    
    Args:
        query: Natural language query
        limit: Maximum results to return
        min_score: Minimum similarity score (0-1)
        constituency: Optional constituency filter
        party: Optional party filter
        
    Returns:
        List of candidate dictionaries with similarity scores
    """
    # Generate embedding for the query
    query_embedding = generate_embedding(query)
    
    # Build the query
    cypher_query = """
    MATCH (c:Candidate)
    WHERE c.embedding IS NOT NULL
    """
    
    params = {
        "query_embedding": query_embedding,
        "limit": limit,
        "min_score": min_score
    }
    
    # Add optional filters
    if constituency:
        cypher_query += """
        AND (c)-[:CONTESTED_IN]->(:Constituency {name: $constituency})
        """
        params["constituency"] = constituency
        
    if party:
        cypher_query += """
        AND (c)-[:MEMBER_OF]->(:Party {name: $party})
        """
        params["party"] = party
    
    # Add vector similarity search
    cypher_query += """
    WITH c, vector.similarity(c.embedding, $query_embedding) AS score
    WHERE score >= $min_score
    RETURN c.name AS name, c.party AS party, c.constituency AS constituency, 
           c.bio AS bio, score
    ORDER BY score DESC
    LIMIT $limit
    """
    
    results = db_manager.run_query(cypher_query, **params)
    return [dict(record) for record in results]