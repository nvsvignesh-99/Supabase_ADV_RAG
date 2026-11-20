import os
from typing import List
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Embeddings (384 dimensions)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Supabase pgvector Retriever compatible with LangChain retriever.invoke()
class SupabaseRetriever(BaseRetriever):
    k: int = 3

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = embeddings.embed_query(query)
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": self.k,
            },
        ).execute()

        return [
            Document(
                page_content=row["content"],
                metadata=row.get("metadata") or {},
            )
            for row in (response.data or [])
        ]

# Create the Retriever
retriever = SupabaseRetriever(k=3)

# Initialize The LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)