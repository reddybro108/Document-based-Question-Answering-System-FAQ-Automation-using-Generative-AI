from src.core.document_loader import load_documents
from src.core.vector_store import create_vector_store, load_vector_store
from src.core.rag_pipeline import build_rag_pipeline

class DocumentQAEngine:
    def __init__(self, doc_path="./data/documents", index_path="faiss_index"):
        chunks = load_documents(doc_path)
        self.vector_store = create_vector_store(chunks, index_path)
        self.qa_chain = build_rag_pipeline(self.vector_store)

    def ask(self, question):
        return self.qa_chain.run(question)
