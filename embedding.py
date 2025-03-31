from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load the Sentence Transformer model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Example texts to store in the vector database
texts = [
    "Artificial Intelligence is transforming the world.",
    "Machine learning helps computers learn from data.",
    "Deep learning is a subset of machine learning.",
    "Python is a popular programming language.",
]

# Convert texts into embeddings and store them in FAISS
vectorstore = FAISS.from_texts(texts, embedding=embeddings)

# Query text to find similar sentences
query = "What is AI?"
similar_docs = vectorstore.similarity_search(query, k=2)

# Print retrieved results
for doc in similar_docs:
    print(doc.page_content)

