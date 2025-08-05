from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def build_rag_pipeline(vector_store, model_name="gpt-3.5-turbo"):
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa_chain
