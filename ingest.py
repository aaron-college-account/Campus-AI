
import pymupdf as fitz
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import faiss


DOCUMENTS_DIR = Path("documents")

def extract_documents():
    pages = []

    for pdf_path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        doc = fitz.open(pdf_path)

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text
                })

        doc.close()

        print(f"Processed: {pdf_path.name}")

    return pages


def chunk_documents(pages, chunk_size=1000, overlap=200):
    chunks = []

    for page in pages:
        text = " ".join(page["text"].split())

        step = chunk_size - overlap

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size]

            if len(chunk_text) < 100:
                continue

            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "text": chunk_text
            })

    return chunks


def create_vector_store(chunks):
    print("\nLoading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index and original chunks
    output_dir = Path("vector_store")
    output_dir.mkdir(exist_ok=True)

    faiss.write_index(
        index,
        str(output_dir / "index.faiss")
    )

    with open(output_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nFAISS index saved with {index.ntotal} vectors.")
    print(f"Embedding dimensions: {dimension}")


if __name__ == "__main__":
    pages = extract_documents()
    chunks = chunk_documents(pages)

    print(f"\nTotal extracted pages: {len(pages)}")
    print(f"Total chunks created: {len(chunks)}")

    create_vector_store(chunks)

