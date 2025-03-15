import os
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
api_key= os.getenv("GOOGLE_API_KEY")


def retrieve_knowledgebase(query: str, k: int= 3) -> str:
    embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store= FAISS.load_local(os.path.join("artifacts/faiss_index"), embeddings)

    retriever= vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs= retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant info found."


RetrievalTool= Tool(
    name="KnowledgeBaseRetriever",
    func=retrieve_knowledgebase,
    description="Retrieves medical information from stored knowledge base."
)
