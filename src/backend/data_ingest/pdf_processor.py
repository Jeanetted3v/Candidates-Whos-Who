import os
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings
from backend.db.neo4j_chunks import add_chunk_to_candidate, add_chunk_to_party, add_chunk_to_constituency

DATA_DIR = "./data/crawled_data/"
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Update: Specify which entity the PDF belongs to (candidate, party, constituency)
def process_pdf(pdf_path, out_dir, candidate_name=None, party_name=None, constituency_name=None, source="pdf"):
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    chunker = SemanticChunker()
    chunks = chunker.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        emb = embeddings.embed_query(text)
        with open(os.path.join(out_dir, f"chunk_{i}.txt"), "w") as f:
            f.write(text)
        # Push to Neo4j
        if candidate_name:
            add_chunk_to_candidate(candidate_name, text, emb, source)
        elif party_name:
            add_chunk_to_party(party_name, text, emb, source)
        elif constituency_name:
            add_chunk_to_constituency(constituency_name, text, emb, source)
        # Otherwise, just store on disk

# Example: You should call process_pdf with the right entity name for each PDF
# For demo, we assume PDFs are named as candidate_<name>.pdf, party_<name>.pdf, or constituency_<name>.pdf

def process_all_pdfs():
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                out_dir = os.path.join(CHUNKS_DIR, os.path.splitext(file)[0])
                os.makedirs(out_dir, exist_ok=True)
                fname = file.lower()
                if fname.startswith("candidate_"):
                    candidate_name = os.path.splitext(file)[0].replace("candidate_", "").replace("_", " ")
                    process_pdf(pdf_path, out_dir, candidate_name=candidate_name)
                elif fname.startswith("party_"):
                    party_name = os.path.splitext(file)[0].replace("party_", "").replace("_", " ")
                    process_pdf(pdf_path, out_dir, party_name=party_name)
                elif fname.startswith("constituency_"):
                    constituency_name = os.path.splitext(file)[0].replace("constituency_", "").replace("_", " ")
                    process_pdf(pdf_path, out_dir, constituency_name=constituency_name)
                else:
                    process_pdf(pdf_path, out_dir)

if __name__ == "__main__":
    process_all_pdfs()
