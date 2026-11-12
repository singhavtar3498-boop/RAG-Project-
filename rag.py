"""
Fully local RAG over web pages (no Ollama).

Stack: sentence-transformers (embeddings), Hugging Face transformers (LLM),
       ChromaDB (vector store), trafilatura (page text extraction).

Usage:
    python rag.py ingest https://example.com/docs https://example.com/faq
    python rag.py ask "How do I reset my password?"
    python rag.py chat            # interactive, loads the LLM only once
"""
import hashlib
import sys

import chromadb
import trafilatura
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"      # ~130 MB, 384-dim embeddings
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"    # ~1 GB; low-memory friendly. Upgrade to 1.5B / 3B if RAM allows
DB_PATH = "./chroma_db"
COLLECTION_NAME = "web_pages"

CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 150     # only used when a single paragraph is too long
TOP_K = 4
MAX_NEW_TOKENS = 300

# bge models work best when queries (not documents) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------- Storage ----------------
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)

# ---------------- Lazy-loaded models ----------------
_embedder = None
_generator = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_generator():
    global _generator
    if _generator is None:
        import torch
        from transformers import pipeline

        print(f"Loading {LLM_MODEL} (first run downloads it)...")
        use_cuda = torch.cuda.is_available()
        _generator = pipeline(
            "text-generation",
            model=LLM_MODEL,
            device=0 if use_cuda else -1,                        # GPU 0 or CPU
            dtype=torch.float16 if use_cuda else torch.bfloat16,  # bfloat16 halves CPU RAM use
        )
    return _generator


def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return get_embedder().encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()


# ---------------- Loading and chunking ----------------
def load_page(url: str) -> str | None:
    html = trafilatura.fetch_url(url)
    return trafilatura.extract(html) if html else None


def chunk(text: str) -> list[str]:
    """Pack paragraphs into chunks of up to CHUNK_SIZE characters."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # Hard-split paragraphs that are longer than a chunk
        while len(para) > CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:CHUNK_SIZE])
            para = para[CHUNK_SIZE - CHUNK_OVERLAP:]

        if len(current) + len(para) + 1 <= CHUNK_SIZE:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)
    return chunks


# ---------------- Ingest ----------------
def ingest(urls: list[str]) -> None:
    for url in urls:
        text = load_page(url)
        if not text:
            print(f"skipped (no content): {url}")
            continue

        pieces = chunk(text)
        if not pieces:
            print(f"skipped (empty after chunking): {url}")
            continue

        # Remove old chunks for this URL so re-ingesting never leaves stale data
        collection.delete(where={"source": url})

        ids = [hashlib.md5(f"{url}-{i}".encode()).hexdigest() for i in range(len(pieces))]
        collection.upsert(
            ids=ids,
            documents=pieces,
            embeddings=embed_documents(pieces),
            metadatas=[{"source": url, "chunk": i} for i in range(len(pieces))],
        )
        print(f"ingested {len(pieces)} chunks from {url}")


# ---------------- Retrieve + generate ----------------
def retrieve(question: str) -> tuple[list[str], list[dict]]:
    results = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=min(TOP_K, collection.count()),
    )
    return results["documents"][0], results["metadatas"][0]


def build_prompt(question: str, docs: list[str]) -> str:
    context = "\n\n---\n\n".join(docs)
    return (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )


def ask(question: str) -> None:
    if collection.count() == 0:
        print("The database is empty. Run: python rag.py ingest <url> first.")
        return

    docs, metas = retrieve(question)
    output = get_generator()(
        [{"role": "user", "content": build_prompt(question, docs)}],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    print(output[0]["generated_text"][-1]["content"].strip())

    print("\nSources:")
    for source in sorted({m["source"] for m in metas}):
        print(f" - {source}")


def chat() -> None:
    print("Interactive mode. Type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", ""}:
            break
        ask(question)


# ---------------- CLI ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"ingest", "ask", "chat"}:
        print(__doc__)
        sys.exit(1)

    command, args = sys.argv[1], sys.argv[2:]
    if command == "ingest":
        if not args:
            print("Give at least one URL.")
            sys.exit(1)
        ingest(args)
    elif command == "ask":
        if not args:
            print("Give a question.")
            sys.exit(1)
        ask(" ".join(args))
    else:
        chat()