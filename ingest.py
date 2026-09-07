import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 1. Native Gemini SDK imports
from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

# Load env variables
load_dotenv()

# Verify API key
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file.")

from tenacity import retry, wait_fixed, wait_exponential, stop_after_attempt

# 2. Custom Embedding Wrapper (Crucial for bypassing rate limits)
class NativeGeminiEmbeddings(Embeddings):
    def __init__(self, model: str = "models/gemini-embedding-2"):
        self.model = model
        self.client = genai.Client()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of documents in batches of 10 natively in 1 API call."""
        embeddings = []
        batch_size = 10  # Keep batch size small to stay under free tier 15 RPM
        
        def _on_retry_sleep(retry_state):
            print(f"Embedding rate limit reached. Waiting 35s before retry attempt {retry_state.attempt_number}...")

        @retry(
            wait=wait_fixed(35),
            stop=stop_after_attempt(15),
            before_sleep=_on_retry_sleep,
            reraise=True
        )
        def _embed_batch_with_retry(contents):
            return self.client.models.embed_content(
                model=self.model,
                contents=contents
            )

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            # Wrap each text in a types.Content object so Gemini knows they are separate texts
            contents = [
                types.Content(parts=[types.Part.from_text(text=t)])
                for t in batch
            ]
            
            print(f"Embedding batch of {len(batch)} items natively...")
            response = _embed_batch_with_retry(contents)
            # Collect the vector values from the response
            for emb in response.embeddings:
                embeddings.append(emb.values)
                
            time.sleep(5)  # Brief pause between native batches
            
        return embeddings

    @retry(
        wait=wait_exponential(multiplier=2, min=15, max=75),
        stop=stop_after_attempt(6),
        reraise=True
    )
    def embed_query(self, text: str) -> list[float]:
        """Embeds a single search query."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return response.embeddings[0].values

import argparse
from legal_splitter import HierarchicalLegalSplitter

def main():
    parser = argparse.ArgumentParser(description="Unified Legal RAG Ingestion Pipeline")
    parser.add_argument("--data", "--dir", dest="data_dir", type=str, default=None,
                        help="Path to legal documents directory (e.g. 'texts_legal' or 'pdfs_leagal')")
    parser.add_argument("--collection", dest="collection_name", type=str, default=None,
                        help="Qdrant collection name (default: auto-derived from corpus)")
    args = parser.parse_args()

    # Determine corpus directory
    data_dir = args.data_dir or os.getenv("CORPUS_DIR")
    if not data_dir:
        if os.path.exists("texts_legal") and any(f.endswith(".txt") for f in os.listdir("texts_legal")):
            data_dir = "texts_legal"
        elif os.path.exists("pdfs_leagal"):
            data_dir = "pdfs_leagal"
        elif os.path.exists("pdfs_legal"):
            data_dir = "pdfs_legal"
        else:
            raise FileNotFoundError("No legal corpus directory found.")

    # 3. Load & Chunk documents hierarchically (PDFs or TXT contracts)
    splitter = HierarchicalLegalSplitter()
    chunks = splitter.split_directory(data_dir)
    print(f"Generated {len(chunks)} structured hierarchical legal chunks from '{data_dir}'.")

    # 4. Initialize our Custom Embeddings Wrapper
    embeddings = NativeGeminiEmbeddings(model="models/gemini-embedding-2")

    # 5. Save to Qdrant Vector Cloud
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    collection_name = args.collection_name or os.getenv("QDRANT_COLLECTION_NAME")
    if not collection_name:
        if "text" in str(data_dir).lower() or "contract" in str(data_dir).lower():
            collection_name = "contracts_legal_rag"
        else:
            collection_name = "hierarchical_legal_rag"

    if qdrant_url and qdrant_api_key:
        print(f"Connecting to Qdrant Cloud at {qdrant_url}...")
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
        )
    else:
        db_path = "./qdrant_db"
        print(f"Connecting to local Qdrant database at {db_path}...")
        client = QdrantClient(path=db_path)

    # Recreate the collection
    if client.collection_exists(collection_name):
        print(f"Deleting old collection '{collection_name}'...")
        client.delete_collection(collection_name)

    print(f"Creating new collection '{collection_name}' with 3072 dimensions...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )

    print(f"Creating payload indexes for strict document scoping...")
    try:
        from qdrant_client.http import models
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.document_name",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.document_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
    except Exception as e:
        print(f"Payload index note: {e}")

    print("Connecting vector store...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    print("Converting text chunks into vectors and uploading in batches...")
    batch_size = 10  # Match our embeddings batch size
    total_chunks = len(chunks)
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Uploading batch {i // batch_size + 1}/{(total_chunks - 1) // batch_size + 1}...")
        
        # Override Qdrant's internal batch size to match ours
        vector_store.add_documents(batch, batch_size=10)
        
        # Sleep for 5 seconds to keep requests per minute strictly below 15
        if i + batch_size < total_chunks:
            print("Sleeping for 5 seconds between batches...")
            time.sleep(5)
        
    print("Database is ready!")

if __name__ == "__main__":
    main()

