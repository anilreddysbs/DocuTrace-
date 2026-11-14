# Root-level app entry point

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docutrace.config import DEFAULT_TOP_K, GEMINI_API_KEY
from docutrace.services.rag_service import RAGService


st.set_page_config(page_title="DocuTrace", page_icon="📚", layout="wide")


@st.cache_resource
def get_rag_service() -> RAGService:
    return RAGService()


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1rem;
        }
        .stChatMessage {
            padding: 0.5rem 0.75rem;
            border-radius: 0.8rem;
        }
        .st-emotion-cache-1d0c3t2 {
            background: #f5f7fb;
        }
        .docutrace-card {
            background: #f5f7fb;
            border: 1px solid #dfe6f1;
            border-radius: 0.8rem;
            padding: 0.9rem 1rem;
            margin: 0.5rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(service: RAGService) -> None:
    st.sidebar.title("📚 DocuTrace")
    st.sidebar.caption("Multi-PDF knowledge assistant")

    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Index Documents", type="primary"):
        if not uploaded_files:
            st.sidebar.warning("Choose at least one PDF to index.")
        else:
            result = service.index_uploaded_files(list(uploaded_files))
            indexed = result.get("indexed", [])
            skipped = result.get("skipped", [])
            if indexed:
                st.sidebar.success(f"Indexed {len(indexed)} document(s).")
            if skipped:
                st.sidebar.info(f"Skipped/failed: {len(skipped)} document(s).")
            st.session_state["indexed_documents"] = service.list_indexed_documents()

    indexed_documents = service.list_indexed_documents()
    st.session_state["indexed_documents"] = indexed_documents
    if indexed_documents:
        selected_docs = st.sidebar.multiselect("Indexed documents", indexed_documents)
        if st.sidebar.button("Remove selected") and selected_docs:
            for doc_id in selected_docs:
                service.remove_document(doc_id)
            st.session_state["indexed_documents"] = service.list_indexed_documents()
            st.sidebar.success(f"Removed {len(selected_docs)} document(s).")

    mode = st.sidebar.selectbox("Retrieval mode", options=["semantic", "hybrid"], index=1)
    st.session_state["retrieval_mode"] = mode
    st.session_state["top_k"] = st.sidebar.slider("Top-k", min_value=1, max_value=10, value=DEFAULT_TOP_K)

    if indexed_documents:
        filter_docs = st.sidebar.multiselect("Optional document filter", indexed_documents)
        st.session_state["document_filter"] = filter_docs
    else:
        st.session_state["document_filter"] = []

    stats = service.vector_store.get_stats()
    st.sidebar.markdown("### Database stats")
    st.sidebar.write(f"Chunks: {stats.get('total_chunks', 0)}")
    st.sidebar.write(f"Documents: {stats.get('total_documents', 0)}")
    st.sidebar.write(f"Gemini API key configured: {'Yes' if GEMINI_API_KEY else 'No'}")


def render_chat(service: RAGService) -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.title("DocuTrace")
    st.caption("Ask questions across all uploaded PDFs and inspect the retrieved evidence.")

    if not st.session_state["messages"]:
        st.markdown(
            """
            <div class="docutrace-card">
            <b>Welcome to DocuTrace.</b><br>
            Upload your PDFs, index them, and ask grounded questions across all sources. Evidence and citations are shown separately from the generated answer.
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                st.caption("Sources: " + ", ".join(c["label"] for c in message["citations"]))

    with st.form("chat_form"):
        question = st.text_input("Ask a question about the indexed documents", key="question_input")
        submit = st.form_submit_button("Ask")

    if submit and question.strip():
        st.session_state["messages"].append({"role": "user", "content": question.strip()})
        answer = service.answer_question(
            question=question.strip(),
            retrieval_mode=st.session_state.get("retrieval_mode", "hybrid"),
            top_k=st.session_state.get("top_k", DEFAULT_TOP_K),
            chat_history=[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state["messages"][:-1]],
            document_filter=st.session_state.get("document_filter", []),
        )

        evidence = answer.get("evidence", [])
        citations = answer.get("citations", [])
        assistant_text = answer.get("answer", "")
        st.session_state["messages"].append({
            "role": "assistant",
            "content": assistant_text,
            "citations": citations,
            "evidence": evidence,
        })

        st.rerun()

    if st.session_state["messages"]:
        if st.button("Clear conversation"):
            st.session_state["messages"] = []
            st.rerun()

    if st.session_state["messages"]:
        last_message = st.session_state["messages"][-1]
        if last_message.get("role") == "assistant" and last_message.get("evidence"):
            st.markdown("### Retrieved evidence")
            for index, item in enumerate(last_message["evidence"], start=1):
                with st.expander(f"[S{index}] {item.get('source')} • Page {item.get('page_number')}"):
                    st.write(item.get("text", ""))
                    st.caption(f"Chunk ID: {item.get('chunk_id')} | Score: {item.get('score', item.get('rrf_score', 0)):.4f}")


def main() -> None:
    render_styles()
    service = get_rag_service()
    render_sidebar(service)
    render_chat(service)


if __name__ == "__main__":
    main()
