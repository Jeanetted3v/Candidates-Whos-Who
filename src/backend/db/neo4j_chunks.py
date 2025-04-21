"""
neo4j_chunks.py
Helper functions for storing and retrieving text/image chunks and their embeddings in Neo4j.
No ChromaDB or external vector DB used. All data is stored in Neo4j nodes and relationships.
(This file was formerly chroma_db.py)
"""
from backend.db.graph_schema import graph
from typing import List, Optional

# Hybrid chunk storage: stores both rule-based metadata and embedding for each chunk
# This enables both deterministic (rule-based) and semantic (embedding) retrieval

def add_chunk_to_candidate(candidate_name: str, text: str, embedding: List[float], source: str = "", 
                          constituency: str = None, party: str = None, image_type: str = None, filename: str = None):
    candidate = graph.nodes.match("Candidate", name=candidate_name).first()
    if candidate:
        chunks = candidate.get("chunks", [])
        chunk_metadata = {
            "text": text,
            "embedding": embedding,
            "source": source,
            "constituency": constituency,
            "party": party,
            "image_type": image_type,
            "filename": filename
        }
        chunks.append(chunk_metadata)
        candidate["chunks"] = chunks
        candidate["embedding"] = embedding  # For vector index
        graph.push(candidate)


def add_chunk_to_party(party_name: str, text: str, embedding: List[float], source: str = "", 
                      image_type: str = None, filename: str = None):
    party = graph.nodes.match("Party", name=party_name).first()
    if party:
        chunks = party.get("chunks", [])
        chunk_metadata = {
            "text": text,
            "embedding": embedding,
            "source": source,
            "image_type": image_type,
            "filename": filename
        }
        chunks.append(chunk_metadata)
        party["chunks"] = chunks
        party["embedding"] = embedding
        graph.push(party)


def add_chunk_to_constituency(constituency_name: str, text: str, embedding: List[float], source: str = "", 
                             party: str = None, image_type: str = None, filename: str = None):
    constituency = graph.nodes.match("Constituency", name=constituency_name).first()
    if constituency:
        chunks = constituency.get("chunks", [])
        chunk_metadata = {
            "text": text,
            "embedding": embedding,
            "source": source,
            "party": party,
            "image_type": image_type,
            "filename": filename
        }
        chunks.append(chunk_metadata)
        constituency["chunks"] = chunks
        constituency["embedding"] = embedding
        graph.push(constituency)

# Helper for extracting rule-based metadata from image filenames
def parse_candidate_filename(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    party_idx = None
    for i, part in enumerate(parts):
        if part in ["pap", "wp", "psp", "sdp", "rp", "sp", "ppp", "dpp", "spp", "nsp", "pv", "red"]:
            party_idx = i
            break
    if party_idx is None or party_idx < 1:
        return None
    constituency = "_".join(parts[:party_idx])
    party = parts[party_idx]
    candidate_name = " ".join(parts[party_idx+1:]).title()
    return constituency, party, candidate_name

# You can add similar helpers for teams, parties, etc.
    constituency = graph.nodes.match("Constituency", name=constituency_name).first()
    if constituency:
        chunks = constituency.get("chunks", [])
        chunks.append({"text": text, "embedding": embedding, "source": source})
        constituency["chunks"] = chunks
        constituency["embedding"] = embedding
        graph.push(constituency)

# Vector search for candidates using Neo4j's vector index (example)
def query_candidates_by_embedding(query_embedding: List[float], top_k: int = 5, region: Optional[str] = None):
    cypher = """
    MATCH (c:Candidate)-[:RUNNING_IN]->(s:Constituency)
    WHERE c.embedding IS NOT NULL {region_clause}
    CALL db.index.vector.queryNodes('candidate_policy_embedding_index', $topK, $embedding) YIELD node, score
    WHERE node = c
    RETURN c.name AS candidate, s.name AS constituency, score
    ORDER BY score DESC
    LIMIT $topK
    """
    region_clause = ""
    params = {"embedding": query_embedding, "topK": top_k}
    if region:
        region_clause = "AND s.region = $region"
        params["region"] = region
    cypher = cypher.format(region_clause=region_clause)
    result = graph.run(cypher, **params)
    return list(result)
