"""
Neo4j database manager for the Election Information App.
Provides a clean interface for database operations using the official Neo4j driver.
To clear the database, run:
python -m src.backend.db.neo4j_manager
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from neo4j import GraphDatabase, Session, Transaction
from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.types import EntityType
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
        self.driver = GraphDatabase.driver(
            SETTINGS.NEO4J_URI,
            auth=(SETTINGS.NEO4J_USER, SETTINGS.NEO4J_PASSWORD)
        )

    async def upsert_embeddings(self, text: str):

        upsert_vectors(
            self.driver, 
            ids=[],
            embedding_property="vectorProperty",
            embeddings=[vector],
            entity_type=EntityType.NODE,
        )





# Clear DB   
if __name__ == "__main__":
    db_manager = Neo4jManager()
    success = db_manager.clear_database()
    
    if success:
        print("Database cleared successfully")
    else:
        print("Failed to clear database")