
import streamlit as st
from pathlib import Path
import logging

st.set_page_config(
    page_title="CampusAI",
    page_icon="🎓",
    layout="wide"
)

# Load backend
@st.cache_resource
def load_backend():
    from rag import generate_answer
    return generate_answer

generate_answer = load_backend()

# Document library
DOCUMENT_DIR = Path(__file__).parent / "documents"
pdf_files = sorted(DOCUMENT_DIR.glob("*.pdf"))

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Styling
st.markdown("""
<style>
.block-container {
    max-width: 1000px;
    padding-top: 3rem;
}

.stChatMessage {
    border-radius: 12px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.2);
}
</style>
""", unsafe_allow_html=True)


# Source display
def display_sources(sources):
    if not sources:
        return

    with st.expander(f"Supporting Sources ({len(sources)})"):
        for i, source in enumerate(sources, start=1):

            st.markdown(
                f"**Source {i}: {source['source']}**"
            )

            st.caption(
                f"Page {source['page']} | "
                f"Similarity: {source['score']:.3f}"
            )

            st.write(source["text"])

            if i < len(sources):
                st.divider()


# Sidebar
with st.sidebar:
    st.title("🎓 CampusAI")
    st.caption("University Policy Assistant")

    st.divider()

    st.subheader("Knowledge Base")
    st.metric("University Documents", len(pdf_files))

    with st.expander("Document Library"):
        for pdf in pdf_files:
            st.write(
                pdf.stem.replace("_SPJ", "").replace("_", " ")
            )

    st.divider()

    st.subheader("System Information")

    st.markdown("""
    **Architecture:** RAG

    **Vector Index:** FAISS

    **Embeddings:** MiniLM-L6-v2

    **Language Model:** GPT-OSS 20B

    **API Provider:** Groq
    """)

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Academic prototype. Verify important decisions "
        "against official university policies."
    )


# Main interface
st.title("🎓 CampusAI")

st.caption("Intelligent University Policy Assistant")

st.write(
    "Ask questions about SP Jain's academic regulations, "
    "university policies and student guidelines."
)

st.divider()


# Display conversation history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))


# User input
question = st.chat_input(
    "Ask CampusAI about university policies..."
)


# Generate answer
if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching university documents..."):

            try:
                history = st.session_state.messages[:-1]
                result = generate_answer(question, history=history)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                display_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception:
                logging.exception("CampusAI error")

                error_message = (
                    "Unable to generate an answer. "
                    "Please try again."
                )

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })
