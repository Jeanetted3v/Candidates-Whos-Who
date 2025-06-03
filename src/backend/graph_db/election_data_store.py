"""
Election data storage module for the Election Information App.

This module provides functions to store extracted election data in Neo4j
and retrieve election information using both rule-based and semantic search.
"""
import logging
from typing import List, Dict, Optional
import json
from datetime import datetime
from src.backend.graph_db.neo4j_manager import db_manager
from src.backend.data_ingest.extract_crawl_model import (
    ExtractedElectionData, 
    ExtractedConstituency,
    ExtractedParty,
    Candidate
)

logger = logging.getLogger(__name__)


# --- Data storage functions ---
def store_extracted_constituency(
    constituency: ExtractedConstituency, tx, source_url: str = None
):
    """Convert and store an ExtractedConstituency to Neo4j nodes and relationships.
    
    Args:
        constituency: The constituency data to store
        tx: The Neo4j transaction object to use for database operations
        source_url: The URL or source where this data was extracted from
    """
    # Create text chunks with source information
    try:
        chunks = []
        if source_url:
            # Create a chunk with the basic constituency information
            chunks.append({
                "text": f"Constituency: {constituency.name}, Type: {constituency.constituency_type}",
                "source": source_url,
                "extraction_date": datetime.now().isoformat()
            })
        # Create constituency query
        chunks_json = json.dumps(chunks or [])
        constituency_query = """
        MERGE (c:Constituency {name: $name})
        SET c.type = $type,
            c.chunks = $chunks,
            c.updated_at = $timestamp
        RETURN c
        """
        # Create the constituency node
        constituency_result = tx.run(
            constituency_query,
            name=constituency.name,
            type=constituency.constituency_type,
            chunks=chunks_json,
            timestamp=datetime.now().isoformat()
        )
        constituency_node = dict(constituency_result.single()["c"])
        
        # Create party nodes and relationships
        for party_name in constituency.contesting_parties:
            # Create party chunks
            party_chunks = []
            if source_url:
                party_chunks.append({
                    "text": f"Party: {party_name}, Contesting in: {constituency.name}",
                    "source": source_url,
                    "extraction_date": datetime.now().isoformat()
                })
                
            # Create party directly with the transaction
            party_query = """
            MERGE (p:Party {name: $name})
            SET p.chunks = $chunks,
                p.updated_at = $timestamp
            RETURN p
            """
            tx.run(
                party_query,
                name=party_name,
                chunks=json.dumps(party_chunks),
                timestamp=datetime.now().isoformat()
            )
            
            # Link party to constituency
            link_query = """
            MATCH (p:Party {name: $party_name})
            MATCH (c:Constituency {name: $constituency_name})
            MERGE (p)-[r:CONTESTED]->(c)
            SET r.year = 2025
            RETURN r
            """
            tx.run(
                link_query,
                party_name=party_name,
                constituency_name=constituency.name
            )

        # Create candidate nodes and relationships
        for candidate in constituency.candidates:
            try:
                logger.info(f"Processing candidate: {candidate.name} for {constituency.name}")
                # Create candidate chunks with source information
                candidate_chunks = []
                if source_url and candidate.bio:
                    candidate_chunks.append({
                        "text": candidate.bio,
                        "source": source_url,
                        "extraction_date": datetime.now().isoformat()
                    })
                    
                # Create candidate directly with the transaction
                candidate_query = """
                MERGE (c:Candidate {name: $name})
                SET c.party = $party,
                    c.constituency = $constituency,
                    c.bio = $bio,
                    c.chunks = $chunks,
                    c.updated_at = $timestamp
                RETURN c
                """
                tx.run(
                    candidate_query,
                    name=candidate.name,
                    party=candidate.party,
                    constituency=candidate.constituency,
                    bio=candidate.bio or "",
                    chunks=json.dumps(candidate_chunks),
                    timestamp=datetime.now().isoformat()
                )
                
                # Link candidate to party directly with the transaction
                link_to_party_query = """
                MATCH (c:Candidate {name: $candidate_name})
                MATCH (p:Party {name: $party_name})
                MERGE (c)-[r:MEMBER_OF]->(p)
                RETURN r
                """
                tx.run(
                    link_to_party_query,
                    candidate_name=candidate.name,
                    party_name=candidate.party
                )
                
                # Link candidate to constituency directly with the transaction
                link_to_constituency_query = """
                MATCH (c:Candidate {name: $candidate_name})
                MATCH (co:Constituency {name: $constituency_name})
                MERGE (c)-[r:CONTESTED_IN]->(co)
                SET r.year = 2025
                RETURN r
                """
                tx.run(
                    link_to_constituency_query,
                    candidate_name=candidate.name,
                    constituency_name=constituency.name
                )
            except Exception as e:
                # Catch and log candidate-specific errors
                logger.error(f"Error processing candidate {candidate.name}: {e}")
                # Continue with next candidate rather than failing the whole constituency
                continue
        
        return constituency_node
    except Exception as e:
        logger.error(f"Error storing constituency data: {e}")
        return None

def store_extracted_election_data(election_data: ExtractedElectionData, source_url: str = None):
    total_stored = 0
    
    # Process constituencies one by one with separate transactions
    for constituency in election_data.constituencies:
        try:
            with db_manager.get_session() as session:
                with session.begin_transaction() as tx:
                    store_extracted_constituency(constituency, tx, source_url)
                    total_stored += 1
        except Exception as e:
            logger.error(f"Failed to store constituency {constituency.name}: {e}")
    
    return total_stored

def check_database_statistics():
    """Check statistics of stored data in Neo4j."""
    stats_query = """
    MATCH (n) 
    RETURN labels(n)[0] AS label, count(*) AS count
    """
    results = db_manager.run_query(stats_query)
    for record in results:
        logger.info(f"{record['label']}: {record['count']} nodes")


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
