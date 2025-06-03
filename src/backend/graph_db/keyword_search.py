import logging
from typing import List, Dict, Any
from src.backend.graph_db.neo4j_manager import db_manager

logger = logging.getLogger(__name__)


def setup_fulltext_indexes():
    """Create all necessary full-text indexes."""
    # Create index for candidates
    candidate_index = """
    CALL db.index.fulltext.createNodeIndex(
        "candidateFulltext",
        ["Candidate"],
        ["name", "bio", "education", "experience", "vision"]
    )
    """
    
    # Create index for parties
    party_index = """
    CALL db.index.fulltext.createNodeIndex(
        "partyFulltext",
        ["Party"],
        ["name", "manifesto", "history", "vision", "policies"]
    )
    """
    
    # Create index for constituencies
    constituency_index = """
    CALL db.index.fulltext.createNodeIndex(
        "constituencyFulltext",
        ["Constituency"],
        ["name", "type", "description"]
    )
    """
    
    try:
        db_manager.run_query(candidate_index)
        db_manager.run_query(party_index)
        db_manager.run_query(constituency_index)
        logger.info("Successfully created all fulltext indexes")
    except Exception as e:
        logger.warning(f"Error creating some fulltext indexes: {e}")

def fulltext_search_candidates(
    query: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for candidates using fulltext index.
    
    Args:
        query: Text to search for
        limit: Maximum number of results to return
        
    Returns:
        List of candidate dictionaries with scores
    """
    try:
        cypher_query = """
        CALL db.index.fulltext.queryNodes("candidateFulltext", $query)
        YIELD node, score
        RETURN node.name AS name, 
               node.party AS party, 
               node.constituency AS constituency, 
               node.bio AS bio,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        results = db_manager.run_query(cypher_query, query=query, limit=limit)
        return [dict(record) for record in results]
    except Exception as e:
        logger.error(f"Error in fulltext search: {e}")
        return []

def fulltext_search_entity(
    query: str, 
    entity_type: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for any entity type using fulltext index.
    
    Args:
        query: Text to search for
        entity_type: Type of entity ("Candidate", "Party", "Constituency")
        limit: Maximum number of results
        
    Returns:
        List of entity dictionaries with scores
    """
    try:
        index_name = f"{entity_type.lower()}Fulltext"
        cypher_query = f"""
        CALL db.index.fulltext.queryNodes("{index_name}", $query)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        results = db_manager.run_query(cypher_query, query=query, limit=limit)
        
        formatted_results = []
        for record in results:
            node_dict = dict(record["node"])
            node_dict["score"] = record["score"]
            node_dict["type"] = entity_type
            formatted_results.append(node_dict)
            
        return formatted_results
    except Exception as e:
        logger.error(f"Error in fulltext search for {entity_type}: {e}")
        return []

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
    """Find all candidates belonging to a specific party."""
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
