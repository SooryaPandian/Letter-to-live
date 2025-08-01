import streamlit as st
import fitz  # PyMuPDF
import spacy
import faiss
# Force CPU-only mode on Windows (faiss-gpu not available)
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

st.title("📚 Letter To Life")
st.markdown("""
Welcome to **Letter To Life**! This app allows you to chat with characters from your favorite books.""")

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

# Helper to track character selection changes
def handle_character_change(new_character):
    if "character_name" in st.session_state and st.session_state.character_name != new_character:
        st.session_state.show_confirm_modal = True
        st.session_state.next_character = new_character
    else:
        st.session_state.character_name = new_character

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
        selected_character = sidebar.selectbox("Choose Character", st.session_state.characters, key="character_select_pre", on_change=None)
        st.session_state.character_name = selected_character

        # Step 3: Chunk text
        chunks = chunk_text(st.session_state.text)
        meta["chunk_count"] = len(chunks)
        progress.progress(0.75, text="Embedding chunks...")
        # Step 4: Embed
        embeddings = embed_texts(chunks)
        dimension = embeddings.shape[1]
        # Use FAISS CPU index (GPU not available on Windows)
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
    selected_character = sidebar.selectbox(
        "Choose Character",
        st.session_state.characters,
        key="character_select",
        index=st.session_state.characters.index(st.session_state.character_name) if st.session_state.character_name in st.session_state.characters else 0,
        on_change=None
    )
    # Detect character change
    if selected_character != st.session_state.character_name:
        st.session_state.show_confirm_modal = True
        st.session_state.next_character = selected_character

# Confirmation modal for character change
if st.session_state.get("show_confirm_modal", False):
    with sidebar:
        st.warning("All previous chats will be cleared. Are you sure you want to change the character?")
        col1, col2 = st.columns(2)
        confirm = col1.button("Yes", key="confirm_change")
        cancel = col2.button("Cancel", key="cancel_change")
        if confirm:
            st.session_state.character_name = st.session_state.next_character
            st.session_state.conversation = []
            st.session_state.chat_history = []
            st.session_state.show_confirm_modal = False
            st.session_state.next_character = None
            st.rerun()
        elif cancel:
            st.session_state.show_confirm_modal = False
            st.session_state.next_character = None
            st.rerun()

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
