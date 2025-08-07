import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pathlib import Path
import hashlib
import time
import asyncio
import nest_asyncio

# Apply nest_asyncio to fix event loop issues
nest_asyncio.apply()

# Configuration
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
VECTORSTORE_DIR = Path("./vectorstore")
VECTORSTORE_DIR.mkdir(exist_ok=True)

# Initialize embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
    request_options={'timeout': 180}
)

# Initialize language model with explicit event loop handling
def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=GOOGLE_API_KEY,
        request_options={'timeout': 180}
    )

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
st.title("📄 PDF Question Answering with Gemini")
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
query = st.text_input("Enter your question")

# Helper functions
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def create_vectorstore_from_docs(docs):
    return FAISS.from_texts(docs, embeddings_model)

def load_chain(vectordb):
    llm = create_llm()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa

# Main processing
if st.button("Get Answer"):
    if uploaded_files and query:
        with st.spinner("Processing..."):
            all_vectorstores = []
            
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                file_hash = generate_pdf_hash(file_bytes)
                uploaded_file.seek(0)  # Reset file pointer
                
                db_path = VECTORSTORE_DIR / f"db_faiss_{file_hash}"
                
                # Load existing or create new vector store
                if db_path.exists():
                    vectordb = FAISS.load_local(str(db_path), embeddings_model)
                else:
                    text = load_pdf_text(uploaded_file)
                    docs = split_text(text)
                    vectordb = create_vectorstore_from_docs(docs)
                    vectordb.save_local(str(db_path))
                
                all_vectorstores.append(vectordb)
            
            # Merge vector stores
            if all_vectorstores:
                combined_vectorstore = all_vectorstores[0]
                for store in all_vectorstores[1:]:
                    combined_vectorstore.merge_from(store)
                
                # Perform QA
                qa_chain = load_chain(combined_vectorstore)
                result = qa_chain({"query": query})
                st.write("**Answer:**", result["result"])
                # Optional: Show sources
                with st.expander("Source Documents"):
                    for doc in result["source_documents"]:
                        st.write(doc.page_content)
                        st.write("---")
    else:
        st.warning("Please upload at least one PDF and enter a question.")
