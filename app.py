import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import chromadb
from dotenv import load_dotenv, find_dotenv
from cryptography.fernet import Fernet

# ------------------------------------------------------------
# ENV + GLOBAL CONFIG
# ------------------------------------------------------------
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # avoid watcher issues on Windows
os.environ["CHROMA_TELEMETRY"] = "FALSE"       # silence telemetry warnings

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "my_book_chunks"
MAX_MEMORY = 6  # chat history limit


# ------------------------------------------------------------
# Load Environment & Groq API Key (with optional decryption)
# ------------------------------------------------------------
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
    except Exception:
        print("Failed to decrypt GROQ key. Falling back to plaintext key if set.")
else:
    print("Plaintext GROQ key will be used (if GROQ_API_KEY is in env).")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="📘 My Baba And I — RAG Assistant",
    layout="centered",
)

st.title("📘 *My Baba And I* — RAG Assistant")
st.markdown(
    "Ask any question from the book and I will answer using the "
    "**exact page(s)** from the text."
)
st.divider()


# ------------------------------------------------------------
# Cached Resources
# ------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_chroma_collection():
    # New Chroma API (v0.4.x)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # optional, but fine
    )
    return collection


embedder = load_embedding_model()
collection = load_chroma_collection()


# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
st.sidebar.header("Settings")
st.sidebar.markdown(f"**Chat Memory Limit:** {MAX_MEMORY} messages")

if st.sidebar.button("🧹 New Conversation"):
    st.session_state["messages"] = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Backend status:**")
try:
    count = collection.count()
    st.sidebar.write(f"✅ Chroma loaded · {count} chunks")
except Exception as e:
    st.sidebar.write("⚠️ Error reading Chroma collection")
    st.sidebar.caption(str(e))


# ------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ------------------------------------------------------------
# Core RAG Function
# ------------------------------------------------------------
def generate_rag_answer(query: str, top_k: int = 3):
    """
    1. Embed the query
    2. Retrieve top_k chunks from Chroma
    3. Build a context string with page numbers
    4. Ask Groq to answer strictly from that context
    """
    # Step 1: Encode Query
    q_embed = embedder.encode(query).tolist()

    # Step 2: Retrieve chunks
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=top_k
    )

    if not results or not results.get("documents") or len(results["documents"][0]) == 0:
        return "I couldn’t find any relevant passages in the book.", [], []

    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    # Step 3: Build Context
    context_parts = []
    for chunk, meta in zip(chunks, metas):
        page = meta.get("book_page", "Unknown")
        context_parts.append(f"--- Page {page} ---\n{chunk}")
    context = "\n\n".join(context_parts)

    # Step 4: Build LLM Prompt
    prompt = f"""
You are a helpful assistant who answers questions using ONLY the context given from a spiritual book.

Your task:
- Read the user's question.
- Read the provided book context.
- Produce a clear, structured answer.
- Use simple language.
- Explicitly cite page numbers in your answer (e.g., "Page 176").
- Summarize the meaning in clear layman's terms.
- If the context does not contain an answer, say you don't know.

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
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = (
            "There was an error while contacting the language model. "
            f"Details: {e}"
        )

    return answer, metas, chunks


# ------------------------------------------------------------
# Chat UI
# ------------------------------------------------------------
st.markdown("## 💬 Chat with the Book")

# Show conversation history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

# User Input
query = st.text_input("Your message:")

# A small UX tweak: allow pressing Enter or clicking the button
send_clicked = st.button("Send")

if send_clicked and query.strip():
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

    # Optional: Show retrieved chunks for transparency
    with st.expander("📄 Show retrieved book pages"):
        if metas and chunks:
            for meta, chunk in zip(metas, chunks):
                page = meta.get("book_page", "Unknown")
                extracted_page = meta.get("extracted_page", "Unknown")
                st.markdown(f"**Book Page:** {page}  ·  **Extracted Page File:** {extracted_page}")
                st.write(chunk)
                st.markdown("---")
        else:
            st.write("No passages retrieved.")

    st.rerun()
