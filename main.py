import os
import openai
import sys
import numpy as np
import requests

from flask import Flask, request, render_template
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

app = Flask(__name__)

# Load PDF
loaders = [
    PyPDFLoader("BeigeBook_20230118.pdf"),
    PyPDFLoader("BeigeBook_20230308.pdf"),
    PyPDFLoader("BeigeBook_20230419.pdf"),
    PyPDFLoader("BeigeBook_20230531.pdf"),
    PyPDFLoader("BeigeBook_20230712.pdf"),
    PyPDFLoader("BeigeBook_20230906.pdf"),
    #PyPDFLoader("BeigeBook_20231018.pdf"),
    #PyPDFLoader("BeigeBook_20231129.pdf") # My current OpenAI subscription does not have enough memory to parse all of the beige books.
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 100
)
splits = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings()

persist_directory = 'docs/chroma/'
vectordb = chroma.Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
# Initialize ChatOpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Define prompt template
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Initialize QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Define route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define route for handling the form submission
@app.route('/submit', methods=['POST'])
def submit():
    question = request.form['question']
    result = qa_chain({"query": question})["result"]
    return render_template('result.html', question=question, result=result)

if __name__ == '__main__':
    app.run(debug=True)