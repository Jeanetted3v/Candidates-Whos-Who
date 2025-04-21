"""
Election data storage module for the Election Information App.

This module provides functions to store extracted election data in Neo4j
and retrieve election information using both rule-based and semantic search.
"""

from typing import List, Dict, Optional
import json
from datetime import datetime
from src.backend.db.neo4j_manager import db_manager
from src.backend.data_ingest.extract_crawl_model import (
    ExtractedElectionData, 
    ExtractedConstituency,
    ExtractedParty,
    Candidate
)

# --- Data storage functions ---
def store_extracted_constituency(constituency: ExtractedConstituency, source_url: str = None):
    """Convert and store an ExtractedConstituency to Neo4j nodes and relationships.
    
    Args:
        constituency: The constituency data to store
        source_url: The URL or source where this data was extracted from
    """
    # Create text chunks with source information
    chunks = []
    if source_url:
        # Create a chunk with the basic constituency information
        chunks.append({
            "text": f"Constituency: {constituency.name}, Type: {constituency.constituency_type}",
            "source": source_url,
            "extraction_date": datetime.now().isoformat()
        })
    
    # Create the constituency node
    constituency_node = db_manager.create_constituency(
        name=constituency.name,
        constituency_type=constituency.constituency_type,
        chunks=chunks
    )
    
    # Create party nodes and relationships
    for party_name in constituency.contesting_parties:
        # Create party chunks with source information
        party_chunks = []
        if source_url:
            party_chunks.append({
                "text": f"Party: {party_name}, Contesting in: {constituency.name}",
                "source": source_url,
                "extraction_date": datetime.now().isoformat()
            })
            
        party_node = db_manager.create_party(name=party_name, chunks=party_chunks)
        db_manager.link_party_to_constituency(party_name, constituency.name)
    
    # Create candidate nodes and relationships
    for candidate in constituency.candidates:
        # Create candidate chunks with source information
        candidate_chunks = []
        if source_url and candidate.bio:
            candidate_chunks.append({
                "text": candidate.bio,
                "source": source_url,
                "extraction_date": datetime.now().isoformat()
            })
            
        candidate_node = db_manager.create_candidate(
            name=candidate.name,
            party=candidate.party,
            constituency=candidate.constituency,
            bio=candidate.bio,
            chunks=candidate_chunks
        )
        db_manager.link_candidate_to_party(candidate.name, candidate.party)
        db_manager.link_candidate_to_constituency(candidate.name, constituency.name)
    
    return constituency_node

def store_extracted_election_data(election_data: ExtractedElectionData, source_url: str = None):
    """Store all extracted election data to Neo4j.
    
    Args:
        election_data: The extracted election data to store
        source_url: The URL or source where this data was extracted from
    """
    for constituency in election_data.constituencies:
        store_extracted_constituency(constituency, source_url)
    
    return len(election_data.constituencies)

# --- Search functions (domain-specific) ---
def search_candidates_by_keyword(keyword: str, limit: int = 10):
    """Search for candidates by keyword in their bio.
    
    Args:
        keyword: Keyword to search for
        limit: Maximum number of results to return
        
    Returns:
        List of candidate dictionaries
    """
    return db_manager.find_candidates_by_text(keyword, limit)

def search_constituencies_by_type(constituency_type: str):
    """Search for constituencies by type (GRC or SMC).
    
    Args:
        constituency_type: Type of constituency to search for
        
    Returns:
        List of constituency dictionaries
    """
    query = """
    MATCH (c:Constituency)
    WHERE c.type = $type
    RETURN c.name AS name, c.type AS type
    """
    
    return db_manager.run_query(query, type=constituency_type)

def find_candidates_by_party(party_name: str):
    """Find all candidates belonging to a specific party.
    
    Args:
        party_name: Name of the party
        
    Returns:
        List of candidate dictionaries
    """
    query = """
    MATCH (c:Candidate)-[:MEMBER_OF]->(p:Party {name: $party_name})
    RETURN c.name AS name, c.constituency AS constituency, c.bio AS bio
    """
    
    return db_manager.run_query(query, party_name=party_name)

def find_parties_in_constituency(constituency_name: str):
    """Find all parties contesting in a specific constituency.
    
    Args:
        constituency_name: Name of the constituency
        
    Returns:
        List of party names
    """
    query = """
    MATCH (p:Party)-[:CONTESTED]->(c:Constituency {name: $constituency_name})
    RETURN p.name AS party_name
    """
    
    result = db_manager.run_query(query, constituency_name=constituency_name)
    return [record["party_name"] for record in result]

# --- Semantic search functions ---
def create_vector_index_for_candidates(dimension: int = 1536):
    """Create a vector index for candidate bio embeddings.
    
    Args:
        dimension: Dimension of the embedding vectors
    """
    query = f"""
    CALL db.index.vector.createNodeIndex(
        'candidate_bio_index',
        'Candidate',
        'embedding',
        {dimension},
        'cosine'
    )
    """
    
    db_manager.run_query(query)

def search_candidates_by_semantic_query(query_embedding: List[float], limit: int = 5):
    """Search for candidates using semantic similarity to a query.
    
    Args:
        query_embedding: Vector embedding of the query
        limit: Maximum number of results to return
        
    Returns:
        List of candidate dictionaries with similarity scores
    """
    query = """
    CALL db.index.vector.queryNodes(
        'candidate_bio_index',
        $limit,
        $embedding
    ) YIELD node, score
    RETURN node.name AS name, 
           node.party AS party, 
           node.constituency AS constituency,
           node.bio AS bio,
           score
    ORDER BY score DESC
    """
    
    return db_manager.run_query(query, limit=limit, embedding=query_embedding)

# --- Database management ---
def close_db_connection():
    """Close the Neo4j database connection."""
    db_manager.close_db()

def init_db_connection():
    """Initialize the Neo4j database connection."""
    db_manager.init_db()
    
def clear_database():
    """Clear all data from the Neo4j database.
    
    This is a destructive operation that will delete all nodes and relationships.
    Use with caution!
    
    Returns:
        True if successful, False otherwise
    """
    return db_manager.clear_database()
