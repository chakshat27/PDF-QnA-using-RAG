import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
from pathlib import Path
import shutil

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize OpenAI model and embeddings
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)


# Prompt template
prompt_template = PromptTemplate.from_template("""
You are a helpful assistant answering based on the content of uploaded PDFs.

Context:
{context}

Question:
{question}

Answer:
""")

# Streamlit UI
st.title("📄 PDF Question Answering with OpenAI")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Enter your question")

def load_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

def create_vector_store(docs):
    db_path = Path("./vectorstore/db_faiss")
    if db_path.exists() and db_path.is_dir():
        shutil.rmtree(db_path)
    vectordb = FAISS.from_documents(docs, embedding)
    vectordb.save_local(str(db_path))
    return vectordb

def load_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="history", input_key="question")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template, "memory": memory}
    )
    return qa

if st.button("Get Answer"):
    if uploaded_file is not None and query:
        text = load_pdf(uploaded_file)
        docs = split_text(text)
        vectordb = create_vector_store(docs)
        qa_chain = load_chain(vectordb)
        result = qa_chain.run(query)
        st.write("**Answer:**", result)
    else:
        st.warning("Upload a PDF and enter a question to proceed.")

