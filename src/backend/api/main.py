import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from backend.non_graph_db.hybrid_retrieval import hybrid_retrieve

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    party: Optional[str] = None
    constituency: Optional[str] = None
    policy_area: Optional[str] = None
    top_k: int = 5

@app.post("/search")
def search(request: QueryRequest):
    filters = {}
    if request.party:
        filters["party"] = request.party
    if request.constituency:
        filters["constituency"] = request.constituency
    if request.policy_area:
        filters["policy_area"] = request.policy_area
    result = hybrid_retrieve(request.query, filters, request.top_k)
    return result

@app.get("/")
def root():
    return {"message": "Election RAG API is running with Neo4j as the single graph+vector database."}
