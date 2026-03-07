import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

st.set_page_config(page_title="Cloud Bot", layout="wide")

# --- 1. SECURE API KEY ---
# In Streamlit Cloud, you'll put this in "Secrets"
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)
    
    # --- 2. KNOWLEDGE BASE ---
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Analyzing document..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            
            # Use Google's embeddings (free tier)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            vector_db = FAISS.from_documents(chunks, embeddings)
            st.sidebar.success("Ready!")

    # --- 3. CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # RAG Search
        context = ""
        if uploaded_file:
            docs = vector_db.similarity_search(prompt, k=3)
            context = "\n".join([d.page_content for d in docs])

        # Generate with Gemini
        model = genai.GenerativeModel('gemini-3-flash')
        full_query = f"Context: {context}\n\nQuestion: {prompt}"
        response = model.generate_content(full_query)
        
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        st.chat_message("assistant").write(response.text)
else:
    st.warning("Please enter your API Key in the sidebar to begin.")