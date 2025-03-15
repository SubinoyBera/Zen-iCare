import os
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


from dotenv import load_dotenv
load_dotenv()

api_key= os.getenv("GOOGLE_API_KEY")

# Extract texts from pdf
def get_pdf_text(pdf_doc):
    text=""
    for pdf in pdf_doc:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    
    return text
            
# Splitting data into small chunks
def get_text_chunks(text):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1000)
    chunks= text_splitter.split_text(text)
    
    return chunks

# Get vector embeddings
def get_vector_store(text_chunks):
    embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store= FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(os.path.join("artifacts/faiss_index"))


def main():
    pdf= [Path('artifacts/medical.pdf')]
    raw_doc= get_pdf_text(pdf)
    text_chunks= get_text_chunks(raw_doc)
    get_vector_store(text_chunks)

if __name__=="__main__":
    main()