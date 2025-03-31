import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load the Sentence Transformer model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to create FAISS vector store
def create_vector_store(text):
    text_chunks = text.split("\n")  # Splitting text into chunks (simplified)
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

# Streamlit UI
st.title("Chat with Your PDFs 📄🤖")
st.sidebar.header("Upload PDFs")
uploaded_pdfs = st.sidebar.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_pdfs:
    with st.spinner("Processing PDFs..."):
        raw_text = extract_text_from_pdfs(uploaded_pdfs)
        vectorstore = create_vector_store(raw_text)
        st.success("PDFs processed! Ask your question below.")

    user_query = st.text_input("Ask a question based on your PDF:")
    if user_query:
        similar_docs = vectorstore.similarity_search(user_query, k=3)
        st.subheader("Relevant Text:")
        for i, doc in enumerate(similar_docs):
            st.write(f"**Match {i+1}:** {doc.page_content}")
