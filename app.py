import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from streamlit_TTS import text_to_speech
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ultimate Personal AI", page_icon="🧠", layout="wide")

# --- SIDEBAR: SETTINGS & KNOWLEDGE ---
with st.sidebar:
    st.title("⚙️ Configuration")
    gemini_key = st.text_input("Gemini API Key", type="password")
    openai_key = st.text_input("OpenAI Key (for Voice)", type="password")
    
    st.divider()
    
    st.header("🎭 Personality")
    persona = st.text_area("Define the bot's character:", 
                          value="You are a helpful, witty assistant.")
    
    st.divider()
    
    st.header("📂 Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- CORE LOGIC ---
if gemini_key:
    # Initialize Google AI
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-3-flash-preview', system_instruction=persona)
    
    # Process PDF for RAG
    if uploaded_file and "vector_db" not in st.session_state:
        with st.status("Reading your document..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader("temp.pdf")
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=gemini_key)
            st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)
            st.success("Knowledge loaded!")

    # --- CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- INPUT HANDLING (Voice or Text) ---
    user_input = None
    
    # Voice Input Widget
    audio_data = st.audio_input("Tap to speak")
    if audio_data and openai_key:
        client = OpenAI(api_key=openai_key)
        with st.spinner("Listening..."):
            transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_data)
            user_input = transcript.text
    
    # Text Input (if no voice recorded)
    if not user_input:
        user_input = st.chat_input("Message your AI...")

    # --- GENERATION ---
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # Get context from PDF if available
            context = ""
            if "vector_db" in st.session_state:
                docs = st.session_state.vector_db.similarity_search(user_input, k=3)
                context = "\n".join([d.page_content for d in docs])
            
            # Formulate prompt and get response
            full_prompt = f"Context: {context}\n\nQuestion: {user_input}"
            response = model.generate_content(full_prompt)
            bot_text = response.text
            
            st.markdown(bot_text)
            
            # Voice Output
            text_to_speech(bot_text, language='en')
            
        st.session_state.messages.append({"role": "assistant", "content": bot_text})

else:
    st.warning("Please enter your Gemini API Key in the sidebar to start.")