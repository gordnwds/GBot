import streamlit as st
from google import genai
from openai import OpenAI
from gtts import gTTS
import io
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- PAGE SETUP ---
st.set_page_config(page_title="Gemini 3 Personal Assistant", page_icon="⚡", layout="wide")

# --- SIDEBAR: KEYS & PERSONA ---
with st.sidebar:
    st.title("🤖 Bot Settings")
    gemini_key = st.text_input("Gemini API Key", type="password")
    openai_key = st.text_input("OpenAI Key (Whisper)", type="password")
    
    st.divider()
    persona = st.text_area("System Persona", 
                          value="You are a brilliant AI assistant. Use provided context to be precise.")
    
    st.divider()
    uploaded_file = st.file_uploader("Upload PDF Knowledge", type="pdf")

# --- INITIALIZATION ---
if gemini_key:
    # Initialize the New 2026 Google GenAI Client
    client = genai.Client(api_key=gemini_key)
    
    # Process RAG (Memory)
    if uploaded_file and "vector_db" not in st.session_state:
        with st.status("Indexing Knowledge..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
            
            # Use the NEW stable embedding model name
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001", 
                google_api_key=gemini_key
            )
            st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)
            st.success("PDF Knowledge Active!")

    # --- CHAT UI ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- INPUT: VOICE OR TEXT ---
    user_query = None
    
    # 2026 Native Audio Input
    audio_input = st.audio_input("Speak to your bot")
    if audio_input and openai_key:
        o_client = OpenAI(api_key=openai_key)
        with st.spinner("Transcribing..."):
            transcript = o_client.audio.transcriptions.create(model="whisper-1", file=audio_input)
            user_query = transcript.text
    
    if not user_query:
        user_query = st.chat_input("Ask a question...")

    # --- GENERATION ---
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            # Pull context from PDF
            context = ""
            if "vector_db" in st.session_state:
                search_results = st.session_state.vector_db.similarity_search(user_query, k=3)
                context = "\n".join([r.page_content for r in search_results])

            # Generate response with Gemini 3 Flash
            full_prompt = f"SYSTEM: {persona}\nCONTEXT: {context}\nUSER: {user_query}"
            
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=full_prompt
            )
            bot_text = response.text
            st.markdown(bot_text)

            # Voice Output (Stable gTTS Method)
            tts = gTTS(text=bot_text, lang='en')
            audio_mem = io.BytesIO()
            tts.write_to_fp(audio_mem)
            st.audio(audio_mem, format="audio/mp3", autoplay=True)

        st.session_state.messages.append({"role": "assistant", "content": bot_text})
else:
    st.info("Please enter your Gemini API key in the sidebar to wake up the bot.")