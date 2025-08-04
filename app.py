import os
import shutil
import requests
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from huggingface_hub import configure_http_backend

from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ---- SSL fix for huggingface ----
# def backend_factory() -> requests.Session:
#     session = requests.Session()
#     session.verify = False
#     return session

# configure_http_backend(backend_factory=backend_factory)

# # ---- Load API key from .env or Streamlit secrets ----
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

# # ---- Initialize models ----
# embeddings_model = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=GOOGLE_API_KEY
# )

# model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

# fixing the SSL issue
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# set GEMINI key
os.environ["GOOGLE_API_KEY"] = " "

# embeddings and model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


# ---- Prompt template ----
prompt_template = PromptTemplate.from_template(
    """
    You are an intelligent assistant designed to answer questions based on the content of uploaded PDF documents.
    Please provide accurate and helpful answers to the questions asked, using the context provided from the documents.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
)
prompt = PromptTemplate(
    template=prompt_template.template,
    input_variables=["history", "context", "question"]
)

# ---- Streamlit UI ----
st.set_page_config(page_title="PDF Q&A with Gemini")
st.title("📄 PDF Question Answering with Gemini")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Ask a question about the document:")

# ---- Helper functions ----
def load_data(uploaded_file):
    os.makedirs("./data", exist_ok=True)
    save_path = os.path.join("./data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFDirectoryLoader("./data/")
    docs = loader.load()
    return docs

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=64)
    return text_splitter.split_documents(docs)

def store_VDB(texts):
    db_path = Path("./vectorstores/db_faiss")
    if db_path.exists() and db_path.is_dir():
        shutil.rmtree(db_path)

    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings_model)
    vectorstore.save_local(str(db_path))

    db = FAISS.load_local(str(db_path), embeddings_model, allow_dangerous_deserialization=True)
    memory = ConversationBufferMemory(memory_key="history", input_key="question")

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"verbose": True, "prompt": prompt, "memory": memory}
    )
    return qa_chain

def query_qa_chain(qa_chain, query):
    response = qa_chain({"query": query})
    return response["result"]

# ---- Run main logic ----
if st.button("Get Answer"):
    if uploaded_file and query:
        docs = load_data(uploaded_file)
        texts = split_text(docs)
        qa_chain = store_VDB(texts)
        answer = query_qa_chain(qa_chain, query)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please upload a PDF and enter a query.")


