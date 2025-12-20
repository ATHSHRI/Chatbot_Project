# app.py
import uuid
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import logging
from dotenv import load_dotenv
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import pdfplumber
from PIL import Image    
import pytesseract

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfminer").disabled = True

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from chromadb.config import Settings
from openai import OpenAI



load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "my_knowledge"

if not OPENROUTER_KEY:
    st.error("Set OPENROUTER_API_KEY in .env and restart.")
    st.stop()

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        coll = client.get_collection(COLLECTION_NAME)
    except Exception:
        coll = client.get_or_create_collection(COLLECTION_NAME)
    return coll

@st.cache_resource
def get_openrouter_client():
    # OpenRouter is OpenAI-compatible; use openai client with base_url override.
    client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
    return client

st.set_page_config(page_title="PDF + Links Chat (OpenRouter)", page_icon="🤖")
st.title("📚 Chat over your PDFs & Links (OpenRouter)")

embedder = get_embedder()
collection = get_chroma_collection()
client = get_openrouter_client()

# ---------------------------------------------------------
# 1️⃣ Define process_uploaded_files BEFORE loading local data
# ---------------------------------------------------------
def process_uploaded_files(uploaded_files):
    for file in uploaded_files:
        content = ""

        try:
            if file.name.lower().endswith(".pdf"):
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        content += page.extract_text() or ""

            elif file.name.lower().endswith((".png", ".jpg", ".jpeg")):
                image = Image.open(file)
                content = pytesseract.image_to_string(image)

            else:
                content = file.read().decode("utf-8")

        except Exception as e:
            st.warning(f"Could not read {file.name}: {e}")
            continue

        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        embeddings = embedder.encode(chunks)

        ids = [str(uuid.uuid4()) for _ in chunks]

        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=[{"source": file.name} for _ in chunks],
            embeddings=[e.tolist() for e in embeddings]
        )

# ---------------------------------------------------------
# 2️⃣ NOW load local files (safe, process_uploaded_files exists)
# ---------------------------------------------------------
LOCAL_DATA_DIR = "./data/pdfs"

def load_local_data():
    if not os.path.exists(LOCAL_DATA_DIR):
        return

    local_files = []
    for fname in os.listdir(LOCAL_DATA_DIR):
        path = os.path.join(LOCAL_DATA_DIR, fname)

        if os.path.isfile(path) and fname.lower().endswith((".pdf", ".txt", ".jpg", ".jpeg", ".png")):
            local_files.append(open(path, "rb"))

    if local_files:
        st.info(f"Loading {len(local_files)} local file(s)...")
        process_uploaded_files(local_files)
        st.success("Local files added to the knowledge base!")

load_local_data()
# ---------------------------------------------------------



uploaded_files = st.file_uploader(
    "Upload PDF, Text, or Image files",
    type=["pdf", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="unique_file_uploader"
)


if uploaded_files:
    with st.spinner("Processing files..."):
        process_uploaded_files(uploaded_files)
    st.success(f"Processed {len(uploaded_files)} file(s) and added to knowledge base.")

if "history" not in st.session_state:
    st.session_state.history = []

def generate_answer(question: str):
    # 1) embed query
    q_emb = embedder.encode([question])[0].tolist()

    # 2) query chroma
    res = collection.query(query_embeddings=[q_emb], n_results=4, include=["documents","metadatas","distances"])
    docs = []
    metadatas = []
    if res and "documents" in res:
        docs = res["documents"][0] if len(res["documents"])>0 else []
        metadatas = res["metadatas"][0] if len(res["metadatas"])>0 else []

    # 3) build context
    context_parts = []
    for d, m in zip(docs, metadatas):
        src = m.get("source", "unknown")
        context_parts.append(f"[{src}] {d}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

    system_prompt = "You are a helpful assistant. Use the context below to answer the user's question. If the answer is NOT in the context, say you don't know rather than making things up."
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer concisely. Also mention the source(s) in square brackets for any factual claims."

    # 4) call OpenRouter via OpenAI-compatible client
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",  # change model if you prefer; check OpenRouter models list
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=700,
        temperature=0.0,
    )

    # extract text (OpenAI-style response)
    try:
        answer = response.choices[0].message.content
    except Exception:
        # fallback
        answer = str(response)

    return answer, docs, metadatas

with st.form("ask_form", clear_on_submit=False):
    q = st.text_input("Ask a question about your uploaded PDFs / links:")
    submitted = st.form_submit_button("Ask")
    if submitted and q:
        st.session_state.history.append({"role":"user","text":q})
        with st.spinner("Thinking..."):
            answer, docs, metadatas = generate_answer(q)
        st.session_state.history.append({"role":"assistant","text":answer})
        

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])

