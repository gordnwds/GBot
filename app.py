import streamlit as st
from google import genai
from google.cloud import speech
from gtts import gTTS
import os
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. AUTHENTICATION ---
# For Local: Point to your JSON file
# For Cloud: You'll paste the JSON content into Streamlit Secrets
if "google_creds" in st.secrets:
    # This part is for when you deploy to Streamlit Cloud
    creds_dict = json.loads(st.secrets["google_creds"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_creds.json"
    with open("google_creds.json", "w") as f:
        json.dump(creds_dict, f)

def transcribe_with_google(audio_bytes):
    client = speech.SpeechClient()
    
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000, # Match st.audio_input default
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    
    for result in response.results:
        return result.alternatives[0].transcript
    return None

# --- PAGE SETUP ---
st.set_page_config(page_title="Gemini 3 Personal Assistant", page_icon="⚡", layout="wide")

# --- SIDEBAR: KEYS & PERSONA ---
with st.sidebar:
    st.title("🤖 Bot Settings")
    gemini_key = st.text_input("Gemini API Key", type="password")
    openai_key = st.text_input("OpenAI Key (Whisper)", type="password")
    
    st.divider()
    auto_play_voice = st.checkbox("Autoplay Bot Voice", value=False)
    
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
    audio_data = st.audio_input("Speak to your bot")

if audio_data:
    with st.spinner("Google is listening..."):
        # Google needs the raw bytes
        user_query = transcribe_with_google(audio_data.getvalue())
    
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
            st.audio(audio_mem, format="audio/mp3", autoplay=auto_play_voice)

        st.session_state.messages.append({"role": "assistant", "content": bot_text})
else:
    st.info("Please enter your Gemini API key in the sidebar to wake up the bot.")