import os
from langchain.tools import Tool
from langchain.vectorstores import FAISS

vector_store= FAISS.load_local(os.path.join("artifacts/faiss_index"), embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": k})
docs = retriever.get_relevant_documents(query)
return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant info found."

MedicalPDFRetrievalTool = Tool(
    name="MedicalPDFRetrieval",
    func=retrieve_medical_info,
    description="Retrieves medical information from stored PDF knowledge base."
)
