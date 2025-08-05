from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def create_vector_store(chunks, save_path="faiss_index"):
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embedding)
    vector_store.save_local(save_path)
    return vector_store

def load_vector_store(path="faiss_index"):
    embedding = OpenAIEmbeddings()
    return FAISS.load_local(path, embedding)
