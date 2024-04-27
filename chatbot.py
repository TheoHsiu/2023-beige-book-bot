import os
import openai
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, render_template
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import chroma
from langchain_openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

app = Flask(__name__)

persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()

vectordb = chroma.Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer concise, averaging around 3-4 sentences.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 4, 'fetch_k': 50}
),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

# Define route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define route for handling the form submission
@app.route('/submit', methods=['POST'])
def submit():
    question = request.form['question']
    result = qa_chain({"query": question})
    answer = result["result"]
    context = result["source_documents"]
    pretty_print_docs(context)
    return render_template('result.html', question=question, result=answer)

if __name__ == '__main__':
    app.run(debug=True)
