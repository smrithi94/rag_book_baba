import os
import shutil
from sentence_transformers import SentenceTransformer
import chromadb

BOOK_OFFSET = 17
PAGES_FOLDER = "extracted_pages"
DB_FOLDER = "chroma_db"
COLLECTION_NAME = "my_book_chunks"

# Only rebuild if DB is missing
if not os.path.exists(DB_FOLDER) or len(os.listdir(DB_FOLDER)) == 0:
    print("Chroma DB missing. Rebuilding...")

    # Delete old folder safely
    if os.path.exists(DB_FOLDER):
        shutil.rmtree(DB_FOLDER)

    # Load text chunks
    def chunk_per_page(folder=PAGES_FOLDER):
        chunks = []
        chunk_id = 1

        if not os.path.exists(folder):
            print("ERROR: extracted_pages folder not found!")
            return []

        filenames = sorted(
            [f for f in os.listdir(folder) if f.endswith(".txt")],
            key=lambda fname: int(fname.replace(".txt", ""))
        )

        for fname in filenames:
            page_num = int(fname.replace(".txt", ""))
            book_page = page_num - BOOK_OFFSET

            if book_page < 1:
                continue

            file_path = os.path.join(folder, fname)
            with open(file_path, encoding="utf-8") as f:
                text = f.read().strip()

            chunks.append({
                "chunk_id": chunk_id,
                "extracted_page": page_num,
                "book_page": book_page,
                "content": text
            })

            chunk_id += 1

        return chunks

    chunks = chunk_per_page()
    print(f"Total chunks found: {len(chunks)}")

    # Load local embedding model
    embedder = SentenceTransformer("models/all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_FOLDER)

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    for chunk in chunks:
        cid = f"chunk_{chunk['chunk_id']}"
        emb = embedder.encode(chunk["content"]).tolist()

        collection.add(
            ids=[cid],
            embeddings=[emb],
            documents=[chunk["content"]],
            metadatas=[{
                "extracted_page": chunk["extracted_page"],
                "book_page": chunk["book_page"],
            }]
        )

    print("Rebuild complete. Total vectors:", collection.count())
else:
    print("Chroma DB already exists. Skipping rebuild.")
