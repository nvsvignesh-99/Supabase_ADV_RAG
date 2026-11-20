import os
import tempfile
import uuid
from datetime import datetime, timezone
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

try:
    from src.rag_pipeline import rag_chain
except ModuleNotFoundError:
    from src.rag_pipline import rag_chain

from src.config import supabase
from ingest_supabase import ingest_pdf_to_supabase

st.set_page_config(
    page_title="Company Policy RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# --- Supabase Chat Persistence Helpers ---

def fetch_sessions():
    res = (
        supabase.table("chat_sessions")
        .select("id, title, updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return res.data or []


def fetch_messages(session_id: str):
    res = (
        supabase.table("chat_messages")
        .select("role, content")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )
    return res.data or []


def ensure_session_exists(session_id: str, first_question: str):
    existing = (
        supabase.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .execute()
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    if not existing.data:
        if len(first_question) > 40:
            title = first_question[:40] + "..."
        else:
            title = first_question
        supabase.table("chat_sessions").insert(
            {"id": session_id, "title": title, "updated_at": now_iso}
        ).execute()
    else:
        supabase.table("chat_sessions").update({"updated_at": now_iso}).eq(
            "id", session_id
        ).execute()


def save_message(session_id: str, role: str, content: str):
    supabase.table("chat_messages").insert(
        {"session_id": session_id, "role": role, "content": content}
    ).execute()


def delete_session(session_id: str):
    supabase.table("chat_sessions").delete().eq("id", session_id).execute()


# --- Session State Initialization ---

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Chat Sessions & PDF Uploader ---

with st.sidebar:
    st.title("💬 Conversations")

    if st.button("➕ New Chat", width="stretch", type="primary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    sessions = fetch_sessions()

    if not sessions:
        st.caption("No saved conversations yet.")
    else:
        for sess in sessions:
            col1, col2 = st.columns([0.82, 0.18])
            is_active = sess["id"] == st.session_state.session_id
            if is_active:
                label = f"🟢 {sess['title']}"
            else:
                label = sess["title"]

            if col1.button(label, key=f"open_{sess['id']}", width="stretch"):
                st.session_state.session_id = sess["id"]
                st.session_state.messages = fetch_messages(sess["id"])
                st.rerun()

            if col2.button("🗑️", key=f"del_{sess['id']}", help="Delete chat"):
                delete_session(sess["id"])
                if is_active:
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.messages = []
                st.rerun()

    st.divider()
    st.subheader("📄 Knowledge Base")
    uploaded_pdf = st.file_uploader("Upload a PDF to Supabase", type=["pdf"])
    if uploaded_pdf and st.button("Index PDF", width="stretch"):
        with st.spinner("Chunking & embedding into Supabase..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name
            try:
                count = ingest_pdf_to_supabase(tmp_path)
                st.success(f"Indexed {count} chunks from {uploaded_pdf.name}!")
            finally:
                os.remove(tmp_path)

# --- Main Chat Interface ---

st.title("🤖 Conversational RAG Chatbot")
st.caption("Powered by LangChain, Groq, HuggingFace Embeddings & Supabase pgvector")

for msg in st.session_state.messages:
    if msg["role"] == "human":
        ui_role = "user"
    else:
        ui_role = "assistant"
    with st.chat_message(ui_role):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about company policy..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    lc_history = []
    for m in st.session_state.messages[-10:]:
        if m["role"] == "human":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = rag_chain.invoke(
                {
                    "input": prompt,
                    "chat_history": lc_history,
                }
            )
            if isinstance(result, str):
                answer = result
            else:
                answer = result.get("answer", str(result))
            st.markdown(answer)

    st.session_state.messages.append({"role": "human", "content": prompt})
    st.session_state.messages.append({"role": "ai", "content": answer})

    ensure_session_exists(st.session_state.session_id, prompt)
    save_message(st.session_state.session_id, "human", prompt)
    save_message(st.session_state.session_id, "ai", answer)
    st.rerun()