import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import telegram.request._httpxrequest as httpx_req
import httpx

# Patch httpx client proxies issue
old_build = httpx_req.HTTPXRequest._build_client
def patched_build_client(self):
    kwargs = dict(self._client_kwargs)
    kwargs.pop("proxies", None)
    return httpx.AsyncClient(**kwargs)
httpx_req.HTTPXRequest._build_client = patched_build_client

load_dotenv()

# Load API keys
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "my_knowledge"

# Load embedder + Chroma
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection(COLLECTION_NAME)

# OpenRouter client
llm_client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")

# Function to answer questions
def generate_answer(question: str):
    q_emb = embedder.encode([question])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=3, include=["documents","metadatas"])
    docs = res["documents"][0] if res and "documents" in res else []
    context = "\n\n".join(docs) if docs else "No relevant context found."

    system_prompt = "You are a helpful assistant that uses the given context to answer questions."
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

    resp = llm_client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_prompt}],
        max_tokens=600,
        temperature=0.0,
    )
    return resp.choices[0].message.content

# Telegram Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! Ask me anything about your PDFs/links 📚")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    answer = generate_answer(user_text)
    await update.message.reply_text(answer)

# Main
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Bot is running on Telegram...")
    app.run_polling()

if __name__ == "__main__":
    main()