# Candidates Who'sWho: RAG Election Info App (still in development)
This project, inspired by Singapore's upcoming General Election 2025, aims to make political information more accessible and structured. It begins by crawling data from the websites of political parties, followed by preprocessing and using a large language model (LLM) to extract key details such as constituencies, contesting parties, candidates, and their bios. The structured data is stored in both a graph database and ChromaDB. When users interact through the UI to ask about a specific constituency or candidate, the system performs Retrieval-Augmented Generation (RAG) to fetch relevant information and generate a response. This setup also serves as an experiment to compare retrieval performance and flexibility between a graph-based and non-graph-based database.

## Updates (29 June 2025)
1. Non-Graph retrieval pipeline is ready
    * Using Chainlit for UI. tested ok so far. 
    * Top 1 retreival is very accurate but not the subsequent retreivals. Metadata extraction part require further experimentation
2. Graph retrival pipeline. 
    * Storing in Neo4j of structured data works well


## Features
- Crawl and process election data (web, PDFs, multimodal)
- Graph database Neo4j
- Non-Graph database: ChromaDB

