from backend.db.graph_schema import *
from backend.db.neo4j_chunks import query_candidates_by_embedding
from langchain.embeddings import OpenAIEmbeddings

# Hybrid retrieval: combines graph traversal and semantic search in Neo4j

def hybrid_retrieve(query, filters=None, top_k=5):
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)
    region = None
    party = None
    constituency = None
    policy_area = None
    if filters:
        region = filters.get("region")
        party = filters.get("party")
        constituency = filters.get("constituency")
        policy_area = filters.get("policy_area")

    # 1. Semantic vector search (optionally filtered by region)
    semantic_results = query_candidates_by_embedding(query_embedding, top_k=top_k, region=region)

    # 2. Relationship-based graph traversal (e.g. by party/constituency)
    graph_results = []
    if party and constituency:
        cypher = f"""
        MATCH (c:Candidate)-[:MEMBER_OF]->(p:Party {{name: '{party}'}}),
              (c)-[:RUNNING_IN]->(s:Constituency {{name: '{constituency}'}})
        RETURN c.name, p.name, s.name
        """
        graph_results = list(graph.run(cypher))

    combined = {
        "semantic": semantic_results,
        "graph": graph_results
    }
    return combined
