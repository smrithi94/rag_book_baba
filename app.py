import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
import chromadb
from dotenv import load_dotenv, find_dotenv
from cryptography.fernet import Fernet

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
# ============================================================
# Load environment variables (Groq API Key)
# ============================================================
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

dotenv_path = find_dotenv("keys.env", usecwd=True)
load_dotenv(dotenv_path)

fernet_key = os.getenv("FERNET_KEY")
encrypted_key = os.getenv("ENCRYPTED_GROQ_API_KEY")

if fernet_key and encrypted_key:
    try:
        groq_api_key = Fernet(fernet_key.encode()).decrypt(
            encrypted_key.encode()
        ).decode()
        os.environ["GROQ_API_KEY"] = groq_api_key
    except:
        print("Failed to decrypt GROQ key.")
else:
    print("Plaintext GROQ key will be used.")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================================
# Streamlit Page Config
# ============================================================
st.set_page_config(
    page_title="📘 My Baba And I — RAG Assistant",
    layout="centered"
)

st.title("📘 *My Baba And I* — RAG Assistant")
st.markdown("Ask any question from the book and I will answer using the **exact page** from the text.")
st.divider()

# ============================================================
# Load Embeddings Model
# ============================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

# ============================================================
# Load ChromaDB Collection
# ============================================================
@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection("my_book_chunks")

collection = load_chroma_collection()

# Chat memory limit
MAX_MEMORY = 6
st.sidebar.markdown(f"**Chat Memory Limit:** {MAX_MEMORY} messages")

if st.sidebar.button("🧹 New Conversation"):
    st.session_state["messages"] = []
    st.rerun()

# ============================================================
# Initialize Chat History Container
# ============================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ============================================================
# RAG Function (NO temperature, NO extra params)
# ============================================================
def generate_rag_answer(query, top_k=3):

    # Step 1: Encode Query
    q_embed = embedder.encode(query).tolist()

    # Step 2: Retrieve chunks
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=top_k
    )

    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    # Step 3: Build Context
    context = ""
    for chunk, meta in zip(chunks, metas):
        context += f"\n--- Page {meta['book_page']} ---\n"
        context += chunk + "\n"

    # Step 4: Build LLM Prompt
    prompt = f"""
You are a helpful assistant who answers questions using ONLY the context given from a book.

Your task:
- Read the user's question.
- Read the provided book context.
- Produce a clear, structured answer.
- Use simple language.
- Cite page numbers (e.g., "Page 176").
- Summarize the meaning in clear layman's terms.

DO NOT add anything outside the context.

QUESTION:
{query}

CONTEXT:
{context}

FORMAT:
**Answer**
(Write a clear explanation here.)

**Pages Referenced**
(List the pages used)
"""

    # Step 5: Call Groq LLM (deterministic)
    response = groq_client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return answer, metas, chunks

# ============================================================
# Chat UI
# ============================================================
st.markdown("## 💬 Chat")

# Display History
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

# User Input
query = st.text_input("Your message:")

if st.button("Send") and query.strip():

    # Add user message
    st.session_state["messages"].append({"role": "user", "content": query})

    # Apply Chat Memory Limit
    if len(st.session_state["messages"]) > MAX_MEMORY:
        st.session_state["messages"] = st.session_state["messages"][-MAX_MEMORY:]

    # Generate Answer
    with st.spinner("Thinking..."):
        answer, metas, chunks = generate_rag_answer(query, top_k=3)

    # Add assistant reply
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    # Apply Chat Memory Limit again
    if len(st.session_state["messages"]) > MAX_MEMORY:
        st.session_state["messages"] = st.session_state["messages"][-MAX_MEMORY:]

    st.rerun()
