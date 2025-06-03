"""To run:
python -m src.backend.main.data_ingest_main
"""
import asyncio
from omegaconf import DictConfig
import hydra
import logging
import json
import os
from src.backend.utils.logging import setup_logging
from src.backend.data_ingest.crawl import crawler_main
from src.backend.data_ingest.llm_extract_crawl import llm_extract_main
from src.backend.graph_db.election_data_store import (
    store_extracted_election_data,
    init_db_connection,
    close_db_connection,
    clear_database,
    check_database_statistics
)
from src.backend.graph_db.semantic_search import (
    create_vector_indexes,
    update_candidate_embeddings
)
from src.backend.data_ingest.extract_crawl_model import ExtractedElectionData
from src.backend.graph_db.keyword_search import setup_fulltext_indexes
from src.backend.non_graph_db.data_processor import NonGraphDBDataProcessor

logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging()


async def data_ingest(cfg: DictConfig) -> None:
    if cfg.crawler.crawl_enabled:
        await crawler_main(cfg)
    if cfg.crawler.extract_json_enabled:
        extracted_data = await llm_extract_main(cfg)
    
    latest_file = os.path.join(
        cfg.crawler.extracted_dir,
        f"{cfg.crawler.file_name}_latest.json"
    )
    logger.info(f"Loading extracted data from {latest_file}")
    with open(latest_file, 'r') as f:
        extracted_data = json.load(f)
    
    # Convert the loaded JSON back to our model
    extracted_data = ExtractedElectionData.model_validate(extracted_data)
    logger.info(f"Loaded {len(extracted_data.constituencies)} constituencies")
    
    if cfg.graphdb.enabled:
        try:
            logger.info("Initializing Neo4j connection")
            init_db_connection()
            
            # Clear database if requested
            if cfg.graphdb.clear_before_import:
                logger.warning("Clearing Neo4j db before importing new data")
                if clear_database():
                    logger.info("Successfully cleared Neo4j database")
                else:
                    logger.error("Failed to clear Neo4j database")
            
            # Store the data
            source_url = cfg.crawler.url
            num_stored = store_extracted_election_data(
                extracted_data,
                source_url
            )
            logger.info(f"Stored {num_stored} constituencies in "
                        f"Neo4j database with source: {source_url}")
            check_database_statistics()
            if cfg.graphdb.embedding_model:
                logger.info(f"Creating vector indexes")
                create_vector_indexes()
                logger.info(f"Updating embeddings")
                update_candidate_embeddings()
                logger.info(f"Creating fulltext indexes")
                setup_fulltext_indexes()
            logger.info("Data ingestion process completed successfully.")
        except Exception as e:
            logger.error(f"Failed to store data in Neo4j: {e}")
        finally:
            logger.info("Closing Neo4j connection")
            close_db_connection()

    if cfg.non_graph_db.enabled:
        processor = NonGraphDBDataProcessor(
            cfg,
            persist_directory=cfg.non_graph_db.persist_dir,
        )
        if cfg.non_graph_db.clear_db:
            logger.warning("Clearing ChromaDB before importing new data")
            processor.clear_database()
        if cfg.non_graph_db.delete_db:
            logger.info("Deleting and recreating ChromaDB before ingestion")
            processor.delete_database()
        await processor.embed_election_data(extracted_data)

@hydra.main(
    version_base=None,
    config_path="../../../config",
    config_name="data_ingest")
def main(cfg) -> None:
    logger.info("Starting data ingestion process...")
    asyncio.run(data_ingest(cfg))


if __name__ == "__main__":
    main()