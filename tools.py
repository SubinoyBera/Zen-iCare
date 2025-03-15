import os
from langchain.tools import Tool
from langchain.vectorstores import FAISS

vector_store= FAISS.load_local(os.path.join("artifacts/faiss_index"))

retriever= vector_store.as_