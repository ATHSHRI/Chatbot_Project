# ingest.py
import os
import glob
import uuid
import json
from pathlib import Path
from dotenv import load_dotenv
import pdfplumber
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

load_dotenv()

# Config
DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
URL_FILE = DATA_DIR / "urls.txt"
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "my_knowledge"

# Make sure directories exist
DATA_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(path):
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n".join(texts)

def extract_text_from_url(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "docbot/1.0"})
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    for s in soup(["script", "style", "noscript", "header", "footer", "aside"]):
        s.decompose()
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)  # overlap
    return chunks

def main(overwrite_collection=False):
    # load embedder (local)
    print("Loading local embedding model (sentence-transformers)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small + fast
    print("Model loaded.")

    # load documents (PDFs)
    sources = []
    for pdf_path in sorted(glob.glob(str(PDF_DIR / "*.pdf"))):
        print("Reading", pdf_path)
        txt = extract_text_from_pdf(pdf_path)
        sources.append({"type":"pdf","source":str(pdf_path), "text": txt})

    # load URLs
    if URL_FILE.exists():
        with open(URL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if url:
                    print("Fetching", url)
                    txt = extract_text_from_url(url)
                    sources.append({"type":"url","source":url,"text":txt})

    # prepare chunks + metadata
    all_chunks = []
    metadatas = []
    ids = []
    for src in sources:
        chunks = chunk_text(src["text"], chunk_size=1200, overlap=200)
        for i,chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadatas.append({"source": src["source"], "type": src["type"], "chunk_index": i})
            ids.append(str(uuid.uuid4()))

    if not all_chunks:
        print("No documents found (put PDFs in data/pdfs/ or URLs in data/urls.txt).")
        return

    # compute embeddings in batch
    print(f"Computing embeddings for {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = [e.tolist() for e in embeddings]

    # init chroma persistent client & collection
    print("Storing into Chroma DB at", CHROMA_DB_DIR)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    # replace or get collection
    if overwrite_collection:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Previous collection removed.")
        except Exception:
            pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Add data (ids, documents, metadatas, embeddings)
    collection.add(
        ids=ids,
        documents=all_chunks,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("Ingestion complete. Stored", len(all_chunks), "chunks in collection:", COLLECTION_NAME)
    # optional: dump metadata summary
    with open("chroma_summary.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(all_chunks), "collection": COLLECTION_NAME}, f)
    print("Summary written to chroma_summary.json")

if __name__ == "_main_":
    # pass True if you want to clear the previous collection
    main(overwrite_collection=False)
