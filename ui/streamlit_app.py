"""Streamlit client: all model and RAG work stays behind the FastAPI API."""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080").rstrip("/")
# Streamlit never receives an LLM key; FastAPI remains the single backend boundary.


def _error_message(error: requests.RequestException) -> str:
    """Turn API/network errors into a safe UI message."""
    response = getattr(error, "response", None)
    if response is not None:
        try:
            return response.json().get("detail", "The API request failed.")
        except ValueError:
            pass
    return "Could not reach the API. Check Docker Compose and /api/health/ready."


st.set_page_config(page_title="Document Research Assistant", page_icon="📚", layout="wide")
st.title("📚 Document Research Assistant")
st.caption("Upload a document, wait for indexing, then ask grounded questions. The browser talks only to FastAPI.")

with st.sidebar:
    st.header("Model route")
    provider = st.selectbox("Provider", ["openai", "vllm"], help="OpenAI uses function calling. vLLM uses your configured local or Colab server.")
    allow_fallback = st.checkbox("Allow vLLM → OpenAI fallback", disabled=provider != "vllm")
    st.divider()
    st.header("Document ingestion")
    upload = st.file_uploader("Upload .txt, .md, .docx, or .pdf", type=["txt", "md", "docx", "pdf"])
    if upload and st.button("Queue document"):
        try:
            response = requests.post(f"{API_BASE_URL}/api/documents/ingest", files={"file": (upload.name, upload.getvalue())}, timeout=30)
            response.raise_for_status()
            st.session_state["job_id"] = response.json()["job_id"]
            st.success(f"Queued {upload.name}")
        except requests.RequestException as error:
            st.error(_error_message(error))
    job_id = st.session_state.get("job_id")
    if job_id:
        if st.button("Check indexing status"):
            try:
                job = requests.get(f"{API_BASE_URL}/api/jobs/{job_id}", timeout=15).json()
                st.info(f"Job {job['status']}")
                if job.get("result"):
                    st.success(f"Indexed {job['result'].get('vectors_created', 0)} vectors")
            except requests.RequestException as error:
                st.error(_error_message(error))

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about indexed documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/chat",
                    json={"query": question, "provider": provider, "allow_fallback": allow_fallback},
                    timeout=120,
                )
                response.raise_for_status()
                answer = response.json()
                st.markdown(answer["answer"])
                meta = answer["metadata"]
                # Expose production behavior so a reviewer can see caching and fallback actually occurred.
                st.caption(f"Requested: {meta['requested_provider']} · Used: {meta['actual_provider']} · Cache: {'hit' if meta['cache_hit'] else 'miss'} · Fallback: {'yes' if meta['fallback_used'] else 'no'} · {meta['latency_ms']} ms")
                with st.expander(f"Verified sources ({len(answer['sources'])})"):
                    for source in answer["sources"]:
                        st.markdown(f"**{source['filename']}** — `{source['chunk_id']}`\n\n{source['excerpt']}")
                st.session_state.messages.append({"role": "assistant", "content": answer["answer"]})
            except requests.RequestException as error:
                st.error(_error_message(error))
