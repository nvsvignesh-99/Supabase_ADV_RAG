from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import supabase, embeddings


def ingest_pdf_to_supabase(pdf_path: str) -> int:
    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Generate 384-dim embeddings
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    # 4. Remove previous chunks for this file to avoid duplicates on re-run
    supabase.table("documents").delete().eq("metadata->>source", pdf_path).execute()

    # 5. Insert chunks + embeddings into Supabase
    rows = [
        {
            "content": chunk.page_content,
            "metadata": {**chunk.metadata, "source": pdf_path},
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    supabase.table("documents").insert(rows).execute()
    print(f"Successfully saved {len(rows)} chunks from '{pdf_path}' to Supabase.")
    return len(rows)


if __name__ == "__main__":
    ingest_pdf_to_supabase("data/company_policy.pdf")