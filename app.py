import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from pathlib import Path
import shutil
import asyncio
import hashlib
import time

# Ensure there's an event loop for gRPC async client
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load Google API key from secrets
google_api_key = st.secrets["GOOGLE_API_KEY"]

# Initialize embeddings and language model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=google_api_key
)

# Define prompt
prompt_template = PromptTemplate.from_template("""
You are a helpful assistant answering based on the content of uploaded PDFs.

Context:
{context}

Question:
{question}

Answer:
""")

# Streamlit UI
st.title("📄 PDF Question Answering with Gemini")
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
query = st.text_input("Enter your question")

# Directory to store vector stores
VECTORSTORE_DIR = Path("./vectorstore")
VECTORSTORE_DIR.mkdir(exist_ok=True)

# Helpers
def generate_pdf_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def load_pdf_text(uploaded_file):
    text = ""
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=64)
    return splitter.create_documents([text])

def embed_documents_in_batches(docs, batch_size=2, retries=3):
    texts = [doc.page_content for doc in docs]
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        attempt = 0
        while attempt < retries:
            try:
                batch_embeddings = embeddings_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                st.warning(f"Retry {attempt + 1}/{retries} after error: {e}")
                time.sleep(2)
                attempt += 1
        else:
            st.error("Failed to embed after multiple retries.")
            raise RuntimeError("Embedding failed.")
    return all_embeddings

def create_vectorstore_from_docs(docs):
    embeddings = embed_documents_in_batches(docs)
    return FAISS.from_embeddings(embeddings, docs)

def load_or_create_vectorstore(docs, unique_id):
    db_path = VECTORSTORE_DIR / f"db_faiss_{unique_id}"
    if db_path.exists():
        vectordb = FAISS.load_local(str(db_path), embeddings_model)
    else:
        vectordb = create_vectorstore_from_docs(docs)
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

# Main logic
if st.button("Get Answer"):
    if uploaded_files and query:
        with st.spinner("Processing..."):
            combined_docs = []

            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                file_hash = generate_pdf_hash(file_bytes)
                uploaded_file.seek(0)  # Reset for PDFReader

                db_path = VECTORSTORE_DIR / f"db_faiss_{file_hash}"
                if db_path.exists():
                    vectordb = FAISS.load_local(str(db_path), embeddings_model)
                else:
                    text = load_pdf_text(uploaded_file)
                    docs = split_text(text)
                    vectordb = create_vectorstore_from_docs(docs)
                    vectordb.save_local(str(db_path))

                combined_docs.extend(vectordb.similarity_search(query, k=3))

            temp_vectordb = create_vectorstore_from_docs(combined_docs)
            qa_chain = load_chain(temp_vectordb)
            result = qa_chain.run(query)
            st.write("**Answer:**", result)
    else:
        st.warning("Please upload at least one PDF and enter a question.")


