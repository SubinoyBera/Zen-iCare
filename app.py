import os
from flask import Flask, render_template, jsonify, request

app= Flask(__name__)

load.env()

embeddings= GoogleGenerativeAIEmbeddings(model='', google_api_key=api_key)
vector_store= FAISS.load_local()
docs= vector_store.similarity_search()
