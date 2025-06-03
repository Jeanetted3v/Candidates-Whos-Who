"""
image_ingest.py

Script to process candidate and team images:
- Parses filenames for metadata (constituency, party, candidate name, etc)
- Uses OpenAI GPT-4 Vision API to extract info and generate embeddings
- Stores embeddings and metadata in Neo4j under the correct node
"""
import os
import re
from typing import List
from PIL import Image
import openai
from backend.graph_db.neo4j_chunks import add_chunk_to_candidate, add_chunk_to_party, add_chunk_to_constituency

# Set your OpenAI API key (or use dotenv)
openai.api_key = os.getenv("OPENAI_API_KEY")

IMG_DIR = "./data/crawled_data/images/"
CANDIDATE_DIR = os.path.join(IMG_DIR, "candidates")
TEAM_DIR = os.path.join(IMG_DIR, "teams")
PARTY_DIR = os.path.join(IMG_DIR, "parties")

# Utility to parse candidate image filenames
def parse_candidate_filename(filename):
    # Example: yio_chu_kang_smc_pap_yip_hon_weng.jpg
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    # Find last occurrence of party (e.g., pap, wp, psp, etc.)
    # Assume party is always before candidate name
    # Example: [yio, chu, kang, smc, pap, yip, hon, weng]
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

# Utility to parse team image filenames
def parse_team_filename(filename):
    # Example: yio_chu_kang_smc_pap_team.jpg
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if parts[-1] != "team":
        return None
    party_idx = None
    for i, part in enumerate(parts):
        if part in ["pap", "wp", "psp", "sdp", "rp", "sp", "ppp", "dpp", "spp", "nsp", "pv", "red"]:
            party_idx = i
            break
    if party_idx is None or party_idx < 1:
        return None
    constituency = "_".join(parts[:party_idx])
    party = parts[party_idx]
    return constituency, party

# Use OpenAI Vision API to get an embedding for an image
def get_image_embedding(image_path):
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
    # The following is a placeholder for calling OpenAI's image embedding endpoint
    # Replace with actual API call when available
    response = openai.embeddings.create(
        model="vision-embedding-001",  # Replace with actual model name
        input=img_bytes
    )
    return response["data"][0]["embedding"]

# Main processing script
def process_candidate_images():
    for fname in os.listdir(CANDIDATE_DIR):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            parsed = parse_candidate_filename(fname)
            if not parsed:
                print(f"Skipping {fname} (unable to parse)")
                continue
            constituency, party, candidate_name = parsed
            img_path = os.path.join(CANDIDATE_DIR, fname)
            embedding = get_image_embedding(img_path)
            add_chunk_to_candidate(
                candidate_name=candidate_name,
                text=f"Image: {fname}",
                embedding=embedding,
                source=img_path,
                constituency=constituency,
                party=party,
                image_type="candidate_face",
                filename=fname
            )
            print(f"Processed candidate image: {fname}")

def process_team_images():
    for fname in os.listdir(TEAM_DIR):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            parsed = parse_team_filename(fname)
            if not parsed:
                print(f"Skipping {fname} (unable to parse)")
                continue
            constituency, party = parsed
            img_path = os.path.join(TEAM_DIR, fname)
            embedding = get_image_embedding(img_path)
            add_chunk_to_constituency(
                constituency_name=constituency,
                text=f"Team Image: {fname}",
                embedding=embedding,
                source=img_path,
                party=party,
                image_type="team_photo",
                filename=fname
            )
            print(f"Processed team image: {fname}")

if __name__ == "__main__":
    process_candidate_images()
    process_team_images()
