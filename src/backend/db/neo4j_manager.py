"""
Neo4j database manager for the Election Information App.
Provides a clean interface for database operations using the official Neo4j driver.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from neo4j import GraphDatabase, Session, Transaction
from src.backend.utils.settings import SETTINGS

logger = logging.getLogger(__name__)

class Neo4jManager:
    """Manages Neo4j database connections and operations."""
    
    def __init__(self, uri=None, user=None, password=None):
        """Initialize the Neo4j manager with connection details.
        
        Args:
            uri: Neo4j connection URI (defaults to SETTINGS.NEO4J_URI)
            user: Neo4j username (defaults to SETTINGS.NEO4J_USER)
            password: Neo4j password (defaults to SETTINGS.NEO4J_PASSWORD)
        """
        self.uri = uri or SETTINGS.NEO4J_URI
        self.user = user or SETTINGS.NEO4J_USER
        self.password = password or SETTINGS.NEO4J_PASSWORD
        self.driver = None
        self.init_db()
        
    def init_db(self):
        """Initialize the Neo4j database connection."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # Test the connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info(f"Successfully connected to Neo4j at {self.uri}")
                else:
                    logger.error("Neo4j connection test failed")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def close_db(self):
        """Close the Neo4j database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
            self.driver = None
    
    @contextmanager
    def get_session(self):
        """Get a Neo4j session as a context manager.
        
        Usage:
            with db_manager.get_session() as session:
                session.run("MATCH (n) RETURN count(n)")
        """
        if not self.driver:
            self.init_db()
            
        if not self.driver:
            raise Exception("Failed to establish Neo4j connection")
            
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def run_query(self, query, **params):
        """Run a Cypher query with parameters.
        
        Args:
            query: Cypher query string
            **params: Query parameters
            
        Returns:
            List of records from the query result
        """
        with self.get_session() as session:
            result = session.run(query, **params)
            return list(result)
    
    def create_constituency(self, name: str, constituency_type: str, 
                           chunks: Optional[List[Dict]] = None) -> Dict:
        """Create a Constituency node in Neo4j.
        
        Args:
            name: Name of the constituency
            constituency_type: Type of constituency (e.g., 'GRC' or 'SMC')
            chunks: List of text chunks with embeddings for RAG architecture
        
        Returns:
            Dictionary representation of the created node
        """
        chunks_json = json.dumps(chunks or [])
        
        query = """
        MERGE (c:Constituency {name: $name})
        SET c.type = $type,
            c.chunks = $chunks,
            c.updated_at = $timestamp
        RETURN c
        """
        
        with self.get_session() as session:
            result = session.run(
                query,
                name=name,
                type=constituency_type,
                chunks=chunks_json,
                timestamp=datetime.now().isoformat()
            )
            record = result.single()
            if record:
                # Convert Neo4j node to dictionary
                node = record["c"]
                return dict(node)
            return None
    
    def create_party(self, name: str, manifesto: Optional[str] = None,
                    chunks: Optional[List[Dict]] = None) -> Dict:
        """Create a Party node in Neo4j.
        
        Args:
            name: Name of the party
            manifesto: Party manifesto text
            chunks: List of text chunks with embeddings
            
        Returns:
            Dictionary representation of the created node
        """
        chunks_json = json.dumps(chunks or [])
        
        query = """
        MERGE (p:Party {name: $name})
        SET p.manifesto = $manifesto,
            p.chunks = $chunks,
            p.updated_at = $timestamp
        RETURN p
        """
        
        with self.get_session() as session:
            result = session.run(
                query,
                name=name,
                manifesto=manifesto or "",
                chunks=chunks_json,
                timestamp=datetime.now().isoformat()
            )
            record = result.single()
            if record:
                node = record["p"]
                return dict(node)
            return None
    
    def create_candidate(self, name: str, party: Optional[str] = None,
                        constituency: Optional[str] = None, 
                        bio: Optional[str] = None,
                        chunks: Optional[List[Dict]] = None) -> Dict:
        """Create a Candidate node in Neo4j.
        
        Args:
            name: Name of the candidate
            party: Party affiliation
            constituency: Constituency name
            bio: Biographical information
            chunks: List of text chunks with embeddings
            
        Returns:
            Dictionary representation of the created node
        """
        chunks_json = json.dumps(chunks or [])
        
        query = """
        MERGE (c:Candidate {name: $name})
        SET c.party = $party,
            c.constituency = $constituency,
            c.bio = $bio,
            c.chunks = $chunks,
            c.updated_at = $timestamp
        RETURN c
        """
        
        with self.get_session() as session:
            result = session.run(
                query,
                name=name,
                party=party or "",
                constituency=constituency or "",
                bio=bio or "",
                chunks=chunks_json,
                timestamp=datetime.now().isoformat()
            )
            record = result.single()
            if record:
                node = record["c"]
                return dict(node)
            return None
    
    def link_party_to_constituency(self, party_name: str, constituency_name: str, 
                                 contested_year: int = 2025) -> bool:
        """Create a CONTESTED relationship between a Party and a Constituency.
        
        Args:
            party_name: Name of the party
            constituency_name: Name of the constituency
            contested_year: Year of the election contest
            
        Returns:
            True if relationship was created, False otherwise
        """
        query = """
        MATCH (p:Party {name: $party_name})
        MATCH (c:Constituency {name: $constituency_name})
        MERGE (p)-[r:CONTESTED]->(c)
        SET r.year = $year
        RETURN r
        """
        
        with self.get_session() as session:
            result = session.run(
                query,
                party_name=party_name,
                constituency_name=constituency_name,
                year=contested_year
            )
            return result.single() is not None
    
    def link_candidate_to_party(self, candidate_name: str, party_name: str) -> bool:
        """Create a MEMBER_OF relationship between a Candidate and a Party.
        
        Args:
            candidate_name: Name of the candidate
            party_name: Name of the party
            
        Returns:
            True if relationship was created, False otherwise
        """
        query = """
        MATCH (c:Candidate {name: $candidate_name})
        MATCH (p:Party {name: $party_name})
        MERGE (c)-[r:MEMBER_OF]->(p)
        RETURN r
        """
        
        with self.get_session() as session:
            result = session.run(
                query,
                candidate_name=candidate_name,
                party_name=party_name
            )
            return result.single() is not None
    
    def link_candidate_to_constituency(self, candidate_name: str, 
                                     constituency_name: str,
                                     contested_year: int = 2025) -> bool:
        """Create a CONTESTED_IN relationship between a Candidate and a Constituency.
        
        Args:
            candidate_name: Name of the candidate
            constituency_name: Name of the constituency
            contested_year: Year of the election contest
            
        Returns:
            True if relationship was created, False otherwise
        """
        query = """
        MATCH (c:Candidate {name: $candidate_name})
        MATCH (co:Constituency {name: $constituency_name})
        MERGE (c)-[r:CONTESTED_IN]->(co)
        SET r.year = $year
        RETURN r
        """
        
        with self.get_session() as session:
            result = session.run(
                query,
                candidate_name=candidate_name,
                constituency_name=constituency_name,
                year=contested_year
            )
            return result.single() is not None
    
    def find_candidates_by_text(self, search_text: str, limit: int = 10) -> List[Dict]:
        """Find candidates by searching their bio text.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of candidate dictionaries
        """
        query = """
        MATCH (c:Candidate)
        WHERE c.bio CONTAINS $search_text
        RETURN c.name AS name, c.party AS party, c.constituency AS constituency, 
               c.bio AS bio
        LIMIT $limit
        """
        
        with self.get_session() as session:
            result = session.run(
                query,
                search_text=search_text,
                limit=limit
            )
            return [dict(record) for record in result]

    def clear_database(self) -> bool:
        """Clear all data from the Neo4j database.
        
        This is a destructive operation that will delete all nodes and relationships.
        Use with caution!
        
        Returns:
            True if successful, False otherwise
        """
        logger.warning("Clearing all data from Neo4j database!")
        
        try:
            query = """
            MATCH (n)
            DETACH DELETE n
            """
            
            with self.get_session() as session:
                session.run(query)
            
            logger.info("Successfully cleared all data from Neo4j database")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Neo4j database: {e}")
            return False

# Singleton instance for easy access
db_manager = Neo4jManager()
