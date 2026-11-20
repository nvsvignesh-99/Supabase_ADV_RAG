# 🤖 Advanced Intent-Based Conversational RAG Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **Groq**, **Hugging Face Embeddings**, **Supabase (`pgvector`)**, and **Streamlit**. It combines intent-aware conversational routing with persistent multi-session chat history and live PDF knowledge base indexing.

---

## ✨ Key Features

- **Intent-Based Conversational Routing:** Distinguishes between general conversational queries and knowledge-base questions while preserving multi-turn chat context.
- **Cloud Vector Search (`pgvector`):** Uses Supabase PostgreSQL with `pgvector` and a custom LangChain `BaseRetriever` for fast cosine-similarity retrieval.
- **Local 384-Dim Embeddings:** Generates dense vector embeddings locally using `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face.
- **High-Speed LLM Inference:** Powered by Groq API via `langchain-groq`.
- **Persistent Multi-Session Chat History:** Stores conversations and messages in Supabase (`chat_sessions` and `chat_messages`) so you can create, revisit, or delete chats anytime.
- **Live PDF Upload & Indexing:** Upload new PDF documents directly from the Streamlit sidebar to chunk, embed, and index them into Supabase on the fly.

---

## 🏗️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Orchestration** | LangChain (`langchain`, `langchain-core`, `langchain-community`) |
| **LLM Provider** | Groq (`langchain-groq`) |
| **Embeddings** | Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) |
| **Vector Store & Database** | Supabase PostgreSQL + `pgvector` |
| **Document Loader & Splitter** | `PyPDFLoader` (`pypdf`) & `RecursiveCharacterTextSplitter` |
| **Frontend UI** | Streamlit |

---

## 📁 Project Structure

```text
Adv_RAG/
├── data/
│   └── company_policy.pdf       # Default source PDF for knowledge base ingestion
├── src/
│   ├── config.py                # Env config, Groq LLM, Hugging Face embeddings & Supabase retriever
│   ├── chains.py                # Intent routing & retrieval-augmented generation chains
│   ├── prompts.py               # System and contextualization prompt templates
│   └── rag_pipeline.py          # Main LangChain Runnable pipeline
├── app.py                       # Streamlit web interface (chat UI, session sidebar, PDF uploader)
├── ingest_supabase.py           # CLI script to chunk, embed, and upload PDFs to Supabase
├── main.py                      # Terminal CLI interface for testing the RAG pipeline
├── requirements.txt             # Python package dependencies
├── .env                         # Environment variables (excluded from Git)
└── .gitignore                   # Git ignore rules
