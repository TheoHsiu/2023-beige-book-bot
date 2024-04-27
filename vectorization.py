import os
import chatbot
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import chroma

_ = load_dotenv(find_dotenv()) # read local .env file

# Load PDF
pdf_files = [
    ("BeigeBook_20230118.pdf", "January 2023"),
    ("BeigeBook_20230308.pdf", "March 2023"),
    ("BeigeBook_20230419.pdf", "April 2023"),
    ("BeigeBook_20230531.pdf", "May 2023"),
    ("BeigeBook_20230712.pdf", "July 2023"),
    ("BeigeBook_20230906.pdf", "September 2023"),
    # ("BeigeBook_20231018.pdf", "October 2023"),
    # ("BeigeBook_20231129.pdf", "November 2023")
]

docs = []
for file, month in pdf_files:
    loader = PyPDFLoader(file)
    pdf_docs = loader.load()

    for doc in pdf_docs:
        doc.metadata['month'] = month
    docs.extend(pdf_docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 100
)
splits = text_splitter.split_documents(docs)

print(len(docs))

embedding = OpenAIEmbeddings()

persist_directory = 'docs/chroma/'
vectordb = chroma.Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
vectordb.persist()
