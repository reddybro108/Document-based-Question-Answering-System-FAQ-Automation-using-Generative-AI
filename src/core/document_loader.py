from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_documents(folder_path):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file in os.listdir(folder_path):
        if file.endswith((".pdf", ".txt", ".docx")):
            loader = UnstructuredFileLoader(os.path.join(folder_path, file))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
    
    return all_chunks
