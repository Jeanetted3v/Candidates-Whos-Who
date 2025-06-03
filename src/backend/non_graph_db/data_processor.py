import json
import os
import logging
import shutil
import re
import asyncio
from omegaconf import DictConfig
from typing import Dict, List, Any
import chromadb
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pydantic_ai import Agent
from src.backend.non_graph_db.metadata_model import EmbeddingMetadata
from src.backend.data_ingest.extract_crawl_model import ExtractedElectionData
from src.backend.utils.settings import SETTINGS

logger = logging.getLogger(__name__)


class NonGraphDBDataProcessor:
    """Process election data into markdown, chunk semantically, and store in ChromaDB with extracted keywords."""
    
    def __init__(self, cfg: DictConfig, persist_directory: str):
        """Initialize the processor."""
        self.cfg = cfg
        self.prompts = cfg.extract_metadata_prompts
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=SETTINGS.OPENAI_API_KEY,
            model_name=cfg.non_graph_db.embedding_model
        )
        self.semantic_chunker = SemanticChunker(
            OpenAIEmbeddings(),
            breakpoint_threshold_type="percentile"
        )
        self.agent = Agent(
            self.cfg.non_graph_db.llm_model,
            result_type=EmbeddingMetadata,
            system_prompt=self.prompts['system_prompt']
        )
        self.collection = self.client.get_or_create_collection(
            name="election_data",
            embedding_function=self.embedding_function,
            metadata={"description": "Election data in semantically chunked Markdown format with structured metadata"}
        )
    def _verify_metadata_in_content(self, content: str, metadata: EmbeddingMetadata) -> EmbeddingMetadata:
        """Verify each metadata item exists in the source content."""
        content_lower = content.lower()
        
        # Verify entity names
        verified_entities = []
        for entity in metadata.entity_names:
            # Check for exact matches or partial matches for longer names
            if (entity.lower() in content_lower or 
                any(part.lower() in content_lower for part in entity.split() if len(part) > 3)):
                verified_entities.append(entity)
        
        # Verify keywords
        verified_keywords = []
        for keyword in metadata.keywords:
            if keyword.lower() in content_lower:
                verified_keywords.append(keyword)
        
        # Create verified metadata
        verified = EmbeddingMetadata(
            primary_category=metadata.primary_category,
            entity_names=verified_entities,
            keywords=verified_keywords,
            related_topics=metadata.related_topics  # Keep related topics as is
        )
        
        return verified
   
    async def _extract_metadata(self, content: str) -> EmbeddingMetadata:
        """Extract metadata using Pydantic AI agent."""
        logger.info("Start extracting metadata")
        try:
            result = await self.agent.run(
                self.prompts['user_prompt'].format(content=content)
            )
            metadata = result.data
            verified_metadata = self._verify_metadata_in_content(content, metadata)
            logger.info(f"Extracted metadata: {verified_metadata}")
            return verified_metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
    
    def convert_to_markdown(self, data: ExtractedElectionData) -> List[Dict[str, Any]]:
        """Convert election data to markdown documents, with one document per constituency."""
        documents = []
        constituencies = data.constituencies
        logger.info(f"Converting {len(constituencies)} constituencies to Markdown")
        
        for constituency in constituencies:
            c_name = constituency.name
            c_type = constituency.constituency_type
            
            if not c_name:
                continue
            
            # Create constituency document with all related information
            md_content = f"# Constituency: {c_name}\n\n"
            md_content += f"**Type**: {c_type}\n\n"
            
            # Get all candidates in this constituency
            candidates = constituency.candidates
            
            # Get all unique parties in this constituency
            parties = set(constituency.contesting_parties)
            
            # Add party information
            if parties:
                md_content += "## Political Parties\n\n"
                for party in sorted(parties):
                    md_content += f"- {party}\n"
                md_content += "\n"
            
            # Add candidates section
            md_content += "## Candidates\n\n"
            
            for candidate in candidates:
                cand_name = candidate.name
                party = candidate.party
                
                if not cand_name:
                    continue
                
                md_content += f"### {cand_name}\n\n"
                md_content += f"**Party**: {party}\n\n"
                
                if candidate.bio:
                    md_content += f"**Biography**:\n\n{candidate.bio}\n\n"
            
            # Create metadata capturing key entities
            metadata = {
                "name": c_name,
                "type": c_type,
                "entity_type": "constituency",
                "id": f"constituency_{c_name.replace(' ', '_').lower()}",
                "candidates": [c.name for c in candidates if c.name],
                "parties": list(parties)
            }
            
            # Add document with content and metadata
            documents.append({
                "content": md_content,
                "metadata": metadata
            })
        
        return documents
    
    async def chunk_and_extract(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Chunk documents semantically and extract metadata."""
        chunked_docs = []
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            try:
                chunks = self.semantic_chunker.create_documents([content])
            except Exception as e:
                logger.error(f"Error chunking document {metadata.get('id')}: {e}")
            # Process each chunk and extract metadata asynchronously
            tasks = []
            for i, chunk in enumerate(chunks):
                chunk_content = chunk.page_content if hasattr(chunk, 'page_content') else chunk
                tasks.append(self._process_chunk(chunk_content, metadata, i, len(chunks)))
            # Wait for all chunk processing to complete
            chunk_results = await asyncio.gather(*tasks)
            chunked_docs.extend(chunk_results)
            logger.info(f"Created {len(chunks)} chunks with extracted metadata for {metadata.get('id')}")
        return chunked_docs
    
    async def _process_chunk(
        self,
        content: str,
        base_metadata: Dict[str, Any], 
        chunk_id: int,
        total_chunks: int
    ) -> Dict[str, Any]:
        """Process a single chunk by extracting metadata.
        
        Args:
            content: Chunk text content
            base_metadata: Base metadata from the document
            chunk_id: Index of this chunk
            total_chunks: Total number of chunks
            
        Returns:
            Processed chunk with enhanced metadata
        """
        extracted_metadata = await self._extract_metadata(content)
        metadata_dict = extracted_metadata.model_dump()
        enhanced_metadata = {
            **base_metadata,
            **metadata_dict,
            "chunk_id": chunk_id,
            "total_chunks": total_chunks,
        }
        return {
            "content": content,
            "metadata": enhanced_metadata
        }
    
    def clear_database(self):
        """Clear all data from the ChromaDB collection."""
        logger.info("Clearing ChromaDB collection")
        try:
            # Get all document IDs first
            all_docs = self.collection.get()
            
            if all_docs and all_docs["ids"]:
                # Delete using the IDs
                self.collection.delete(ids=all_docs["ids"])
                logger.info(f"Successfully cleared {len(all_docs['ids'])} documents from ChromaDB collection")
                return True
            else:
                logger.info("Collection is already empty, nothing to clear")
                return True
        except Exception as e:
            logger.error(f"Error clearing ChromaDB collection: {e}")
    
    def delete_database(self):
        """Delete the entire ChromaDB database directory."""
        logger.info("Deleting ChromaDB directory")
        try:
            # Get the directory from client
            db_dir = self.client._persist_directory
            
            # Delete the collection first
            self.collection.delete(where={})
            
            # Close the client connection
            self.client = None
            
            # Delete the directory
            shutil.rmtree(db_dir)
            logger.info(f"Successfully deleted ChromaDB directory at {db_dir}")
            
            # Reinitialize the client and collection
            self.client = chromadb.PersistentClient(path=db_dir)
            self.collection = self.client.get_or_create_collection(
                name="election_data",
                embedding_function=self.embedding_function,
                metadata={"description": "Election data in semantically chunked Markdown format with structured metadata"}
            )
            
            return True
        except Exception as e:
            logger.error(f"Error deleting ChromaDB directory: {e}")
            return False
    
    def _sanitize_text(self, text: str) -> str:
        """Clean text of control characters and non-standard quotes."""
        if not text:
            return text
        text = text.replace('\x03', ' ')  # ETX character
        text = text.replace('\x0093', '"')  # Windows left quote
        text = text.replace('\x0094', '"')  # Windows right quote
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
        return text

    async def embed_election_data(self, extracted_json: dict):
        """Process election data JSON file into ChromaDB.
        
        Args:
            json_file_path: Path to the election data JSON file
        """
        if not isinstance(extracted_json, ExtractedElectionData):
            extracted_json = ExtractedElectionData.model_validate(extracted_json)
        documents = self.convert_to_markdown(extracted_json)
        logger.info(f"Created {len(documents)} from markdown documents")
        chunked_docs = await self.chunk_and_extract(documents)
        logger.info(f"Created {len(chunked_docs)} chunks with extracted metadata")
        
        # Add to ChromaDB in batches
        batch_size = 50
        total_added = 0
        
        for i in range(0, len(chunked_docs), batch_size):
            batch = chunked_docs[i:i+batch_size]
            
            ids = [f"{doc['metadata']['id']}_chunk{doc['metadata']['chunk_id']}" for doc in batch]
            logger.info(f"IDs: {ids}")
            contents = [self._sanitize_text(doc["content"]) for doc in batch]
            logger.info(f"Contents: {contents}")
            logger.info(f"IDs: {ids}")
            processed_metadatas = [
                {
                    key: json.dumps(value) if not isinstance(value, (str, int, float, bool))
                    else value
                    for key, value in doc["metadata"].items()
                    if value is not None
                } for doc in batch
            ]
            try:
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=processed_metadatas
                )
                total_added += len(batch)
                logger.info(f"Added batch of {len(batch)} chunks ({i+1}-{i+len(batch)} of {len(chunked_docs)})")
            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {e}")
                successful = 0
                for j in range(len(ids)):
                    try:
                        self.collection.add(
                            ids=[ids[j]],
                            documents=[contents[j]],
                            metadatas=[processed_metadatas[j]]
                        )
                        successful += 1
                    except Exception as inner_e:
                        logger.error(f"Failed to add document {ids[j]}: {inner_e}")
            
            if successful > 0:
                logger.info(f"Added {successful} documents individually after batch failure")
                total_added += successful
        
        logger.info(f"Successfully added {total_added} chunked documents to ChromaDB")