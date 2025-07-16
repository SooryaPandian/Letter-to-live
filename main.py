import streamlit as st
import fitz  # PyMuPDF
import spacy
import faiss
try:
    import faiss.contrib.torch_utils  # noqa: F401
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
import os
from ollama import Client
from typing import List, Tuple
import numpy as np

# Initialize Ollama client
client = Client()

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Embedding model
EMBED_MODEL = "llama3.2:latest"

# -----------------------------
# Utility Functions
# -----------------------------

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    total_pages = doc.page_count
    progress = st.progress(0, text="Reading PDF pages...")
    for i, page in enumerate(doc):
        text += page.get_text()
        percent = int((i + 1) / total_pages * 100)
        progress.progress((i + 1) / total_pages, text=f"Reading PDF pages... {percent}%")
    progress.empty()
    return text

def extract_characters(text):
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    freq = {}
    for name in names:
        freq[name] = freq.get(name, 0) + 1
    # Return sorted list of unique names
    sorted_names = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [name for name, count in sorted_names if count > 2]

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def embed_texts(texts: List[str], batch_size: int = 8) -> np.ndarray:
    """
    Embeds texts in batches using Ollama. If Ollama does not support batch embedding, this will fallback to single requests.
    """
    embeddings = []
    total = len(texts)
    progress = st.progress(0, text="Embedding chunks...")
    idx = 0
    while idx < total:
        batch = texts[idx:idx+batch_size]
        try:
            # Try batch embedding (if supported by Ollama)
            response = client.embeddings(model=EMBED_MODEL, prompt=batch)
            # If response is a list of embeddings
            if isinstance(response, dict) and "embeddings" in response:
                batch_embeddings = response["embeddings"]
            elif isinstance(response, list):
                batch_embeddings = [r["embedding"] for r in response]
            else:
                # Fallback: single embedding per batch
                batch_embeddings = [response["embedding"] for _ in batch]
        except Exception:
            # Fallback to single requests if batch fails
            batch_embeddings = []
            for t in batch:
                r = client.embeddings(model=EMBED_MODEL, prompt=t)
                batch_embeddings.append(r["embedding"])
        embeddings.extend(batch_embeddings)
        idx += batch_size
        percent = int(min(idx, total) / total * 100)
        progress.progress(min(idx, total) / total, text=f"Embedding chunks... {percent}%")
    progress.empty()
    return np.array(embeddings).astype("float32")

def get_top_k_chunks(question, index, chunk_texts, k=3):
    q_emb = client.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    q_emb = np.array(q_emb).astype("float32").reshape(1, -1)
    D, I = index.search(q_emb, k)
    return [chunk_texts[i] for i in I[0]]

# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title="Book Character Chat", layout="wide")

st.title("📚 Talk to Your Favorite Book Character")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "chunk_texts" not in st.session_state:
    st.session_state.chunk_texts = []

if "character_name" not in st.session_state:
    st.session_state.character_name = ""


# --- PDF Upload and Processing (only once) ---
sidebar = st.sidebar
uploaded_file = sidebar.file_uploader("Upload a Book (PDF)", type="pdf")

if uploaded_file and "pdf_processed" not in st.session_state:
    meta = {}
    progress = sidebar.progress(0, text="Processing PDF...")
    with st.spinner("Processing PDF..."):
        # Step 1: Read PDF
        text = extract_text_from_pdf(uploaded_file)
        meta["word_count"] = len(text.split())
        progress.progress(0.25, text="Extracting characters...")
        # Step 2: Extract characters
        characters = extract_characters(text)
        meta["character_count"] = len(characters)
        if not characters:
            progress.empty()
            sidebar.warning("No characters detected. Try a different book.")
            st.stop()
        st.session_state.characters = characters
        st.session_state.text = text
        sidebar.success("Characters extracted. Select one below.")
        progress.progress(0.5, text="Chunking text...")

        # Character selection (only before embedding)
        st.session_state.character_name = sidebar.selectbox("Choose Character", st.session_state.characters, key="character_select_pre")

        # Step 3: Chunk text
        chunks = chunk_text(st.session_state.text)
        meta["chunk_count"] = len(chunks)
        progress.progress(0.75, text="Embedding chunks...")
        # Step 4: Embed
        embeddings = embed_texts(chunks)
        dimension = embeddings.shape[1]
        # Use FAISS GPU if available
        if GPU_AVAILABLE:
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatL2(dimension)
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        st.session_state.faiss_index = index
        st.session_state.chunk_texts = chunks
        st.session_state.pdf_processed = True
        st.session_state.meta_info = meta
        progress.progress(1.0, text="Ready to chat!")
        progress.empty()

# If already processed, show character select (for re-selection)
elif "pdf_processed" in st.session_state and "characters" in st.session_state:
    st.session_state.character_name = st.selectbox("Choose Character", st.session_state.characters, key="character_select")

# --- Chat Interface ---
if st.session_state.get("faiss_index") and st.session_state.get("character_name"):
    # --- Chat Interface (main area, right side) ---
    # Use st.chat_message and st.chat_input for modern chat UI
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Display chat messages
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for user's message
    prompt = st.chat_input("Type your message...")

    def run_character_llm(user_input, conversation_history):
        # Retrieve relevant context
        retrieved_chunks = get_top_k_chunks(
            user_input,
            st.session_state.faiss_index,
            st.session_state.chunk_texts,
            k=3
        )
        context = "\n\n".join(retrieved_chunks)
        system_prompt = f"""
You are {st.session_state.character_name}, a character from the book.
You will answer questions ONLY using the context provided.
Stay in character, using the same tone and style you used in the story.
If you don't know, say you don't remember.

Context:
{context}
"""
        # Build message history for model
        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_history,
            {"role": "user", "content": user_input}
        ]
        response = client.chat(
            model=EMBED_MODEL,
            messages=messages
        )
        answer = response["message"]["content"]
        return {"role": "assistant", "content": answer}

    if prompt:
        # Add user's message to conversation
        new_message = {"role": "user", "content": prompt}
        st.session_state.conversation.append(new_message)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Waiting for response..."):
            response_message = run_character_llm(prompt, st.session_state.conversation[:-1])
            with st.chat_message("assistant"):
                st.markdown(response_message["content"])
            st.session_state.conversation.append(response_message)
