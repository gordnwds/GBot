import streamlit as st
from google import genai
from google.cloud import speech
from google.oauth2 import service_account
from gtts import gTTS
import io
import json
import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- PAGE SETUP ---
st.set_page_config(page_title="Google-Native AI Bot", page_icon="☁️", layout="wide")

# --- AUTHENTICATION HELPER ---
def get_google_speech_client():
    # If on Streamlit Cloud, use Secrets
    if "google_creds" in st.secrets:
        creds_info = json.loads(st.secrets["google_creds"])
        creds = service_account.Credentials.from_service_account_info(creds_info)
        return speech.SpeechClient(credentials=creds)
    # If local, it looks for the file
    return speech.SpeechClient()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    gemini_key = st.text_input("Gemini API Key", type="password")
    st.info("Note: Google Speech-to-Text uses the JSON key in your Streamlit Secrets.")
    
    st.divider()
    persona = st.text_area("Bot Persona", value="You are a helpful assistant.")
    uploaded_file = st.file_uploader("Upload PDF Knowledge", type="pdf")
    auto_play = st.checkbox("Autoplay Voice", value=False)

# --- APP LOGIC ---
if gemini_key:
    # Initialize Gemini 3
    client = genai.Client(api_key=gemini_key)
    
    # RAG Indexing
    if uploaded_file and "vector_db" not in st.session_state:
        with st.status("Indexing PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader("temp.pdf")
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=gemini_key)
            st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)
            st.success("Knowledge Base Ready!")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- INPUT: GOOGLE SPEECH-TO-TEXT ---
    user_query = None
    audio_input = st.audio_input("Speak to your bot")

    if audio_input:
        with st.spinner("Google Cloud transcribing..."):
            try:
                speech_client = get_google_speech_client()
                audio_content = audio_input.getvalue()
                
                audio = speech.RecognitionAudio(content=audio_content)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    language_code="en-US",
                    enable_automatic_punctuation=True
                )

                response = speech_client.recognize(config=config, audio=audio)
                if response.results:
                    user_query = response.results[0].alternatives[0].transcript
                else:
                    st.error("Google couldn't hear anything. Try speaking louder!")
            except Exception as e:
                st.error(f"Google Speech Error: {e}")

    if not user_query:
        user_query = st.chat_input("Or type here...")

    # --- RESPONSE GENERATION ---
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            context = ""
            if "vector_db" in st.session_state:
                results = st.session_state.vector_db.similarity_search(user_query, k=3)
                context = "\n".join([r.page_content for r in results])

            full_prompt = f"SYSTEM: {persona}\nCONTEXT: {context}\nUSER: {user_query}"
            
            # Using Gemini 3 Flash
            # --- AUTO-RETRY LOGIC FOR 503 ERRORS ---
            bot_text = ""
            for attempt in range(3): # Try up to 3 times
                try:
                    resp = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=full_prompt
                    )
                    bot_text = resp.text
                    break # Success!  Exit the loop
                except Exception as e:
                    if "503" in str(e) or "overloaded" in str(e):
                        st.warning(f"Server busy (Attempt {attempt+1}/3). Retrying...")
                        time.sleep(2 ** attempt) # Wait 1s, then 2s
                    else:
                        st.error(f"Error: {e}")
                        break

        if bot_text:
            st.markdown(bot_text)

            # Voice Out
            tts = gTTS(text=bot_text, lang='en')
            audio_mem = io.BytesIO()
            tts.write_to_fp(audio_mem)
            st.audio(audio_mem, format="audio/mp3", autoplay=auto_play)

        st.session_state.messages.append({"role": "assistant", "content": bot_text})