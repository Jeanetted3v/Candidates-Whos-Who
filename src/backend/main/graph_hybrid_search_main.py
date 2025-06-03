
import asyncio
from typing import List, Dict, Any
from omegaconf import DictConfig
import hydra
import logging
from pydantic_ai import Agent
from src.backend.graph_db.neo4j_manager import db_manager
from src.backend.utils.logging import setup_logging
from src.backend.graph_db.semantic_search import semantic_search_candidates
from src.backend.graph_db.keyword_search import (
    fulltext_search_entity
)

logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging()


class HybridSearch():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.semantic_weight = cfg.get("semantic_weight", 0.7)
        self.fulltext_weight = cfg.get("fulltext_weight", 0.3)
        self.prompts = cfg.hybrid_search.prompts
        self.response_agent = Agent(
            model=cfg.llm_model,
        )

def hybrid_search(
    cfg: DictConfig,
    query: str, 
    entity_types: List[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Enhanced hybrid search across multiple entity types.
    
    Args:
        query: The search query
        entity_types: List of entity types to search ['Candidate', 'Party', 'Constituency']
                      If None, search all types
        limit: Maximum number of results to return
        semantic_weight: Weight for semantic search results (0-1)
        fulltext_weight: Weight for fulltext search results (0-1)
        
    Returns:
        List of result dictionaries with combined scores
    """
    # Default to all entity types if not specified
    if entity_types is None:
        entity_types = ['Candidate', 'Party', 'Constituency']
    semantic_weight = cfg.get("semantic_weight", 0.7)
    fulltext_weight = cfg.get("fulltext_weight", 0.3)
    
    # Get semantic search results for each entity type
    semantic_results = []
    for entity_type in entity_types:
        if entity_type == 'Candidate':
            results = semantic_search_candidates(query, limit=limit)
            for result in results:
                result["type"] = "Candidate"
            semantic_results.extend(results)
        else:
            try:
                # Generate embedding for query
                query_embedding = generate_embedding(query)
                
                # Create query for vector similarity search
                cypher_query = f"""
                MATCH (n:{entity_type})
                WHERE n.embedding IS NOT NULL
                WITH n, vector.similarity(n.embedding, $query_embedding) AS score
                WHERE score >= 0.7
                RETURN n, score
                ORDER BY score DESC
                LIMIT $limit
                """
                results = db_manager.run_query(
                    cypher_query, 
                    query_embedding=query_embedding,
                    limit=limit
                )
                
                # Format results
                for record in results:
                    node_dict = dict(record["n"])
                    node_dict["score"] = record["score"]
                    node_dict["type"] = entity_type
                    semantic_results.append(node_dict)
                    
            except Exception as e:
                logger.error(f"Error in semantic search for {entity_type}: {e}")
    
    # Get fulltext search results for each entity type
    fulltext_results = []
    for entity_type in entity_types:
        try:
            entity_results = fulltext_search_entity(query, entity_type, limit=limit)
            fulltext_results.extend(entity_results)
        except Exception as e:
            logger.error(f"Error in fulltext search for {entity_type}: {e}")
    # Combine results
    combined = {}
    
    # Process semantic results
    for result in semantic_results:
        # Create a unique identifier based on type and name
        entity_type = result.get("type", "Unknown")
        name = result.get("name", "")
        key = f"{entity_type}:{name}"
        
        combined[key] = {
            **result,
            "semantic_score": result.get("score", 0),
            "entity_type": entity_type
        }
    
    # Process fulltext results
    for result in fulltext_results:
        entity_type = result.get("type", "Unknown")
        name = result.get("name", "")
        key = f"{entity_type}:{name}"
        
        if key in combined:
            # Entity exists from semantic search
            combined[key]["fulltext_score"] = result.get("score", 0)
            combined[key]["combined_score"] = (
                combined[key].get("semantic_score", 0) * semantic_weight + 
                result.get("score", 0) * fulltext_weight
            )
        else:
            # New entity from fulltext search
            combined[key] = {
                **result,
                "fulltext_score": result.get("score", 0),
                "semantic_score": 0,
                "entity_type": entity_type,
                "combined_score": result.get("score", 0) * fulltext_weight
            }
    
    # Convert to list, ensure all entries have combined_score
    results = list(combined.values())
    for result in results:
        if "combined_score" not in result:
            result["combined_score"] = result.get("semantic_score", 0) * semantic_weight
    
    # Sort by combined score
    results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    return results[:limit]


@hydra.main(
    version_base=None,
    config_path="../../../config",
    config_name="data_ingest")
def main(cfg) -> None:
    logger.info("Starting hybrid search setup...")
    
    
    # Example search (when running script directly)
    if cfg.get("search_query"):
        results = hybrid_search(
            query=cfg.search_query,
            entity_types=cfg.get("entity_types"),
            limit=cfg.get("limit", 10)
        )
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: {result['name']} ({result['type']}) - Score: {result['score']}")


if __name__ == "__main__":
    main()