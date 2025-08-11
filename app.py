# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from pathlib import Path
# import hashlib
# import time
# import asyncio
# import nest_asyncio
# import logging
# import random

# # Apply nest_asyncio to fix event loop issues
# nest_asyncio.apply()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configuration
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
# VECTORSTORE_DIR = Path("./vectorstore")
# VECTORSTORE_DIR.mkdir(exist_ok=True)

# # Initialize embeddings model with increased timeout
# embeddings_model = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=GOOGLE_API_KEY,
#     request_options={'timeout': 300}  # Increased timeout to 300 seconds
# )

# # Initialize language model with explicit event loop handling
# def create_llm():
#     return ChatGoogleGenerativeAI(
#         model="gemini-pro",
#         google_api_key=GOOGLE_API_KEY,
#         request_options={'timeout': 300}  # Increased timeout to 300 seconds
#     )

# # Prompt template
# prompt_template = PromptTemplate.from_template("""
# You are a helpful assistant answering based on the content of uploaded PDFs.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """)

# # Streamlit UI
# st.title("📄 PDF Question Answering with Gemini")
# uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
# query = st.text_input("Enter your question")

# # Helper functions
# def generate_pdf_hash(file_bytes):
#     return hashlib.md5(file_bytes).hexdigest()

# def load_pdf_text(uploaded_file):
#     text = ""
#     pdf_reader = PdfReader(uploaded_file)
#     for page in pdf_reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text
#     return text

# def split_text(text):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=600,  # Optimized for Google API
#         chunk_overlap=80,
#         length_function=len
#     )
#     return splitter.split_text(text)

# def embed_with_retry(texts, max_retries=5):
#     """Embed documents with robust retry logic"""
#     attempts = 0
#     while attempts < max_retries:
#         try:
#             return embeddings_model.embed_documents(texts)
#         except Exception as e:
#             attempts += 1
#             if attempts < max_retries:
#                 wait_time = (2 ** attempts) + random.uniform(0, 1)  # Exponential backoff with jitter
#                 logger.warning(f"Embedding failed, retry {attempts}/{max_retries} in {wait_time:.1f}s. Error: {str(e)}")
#                 time.sleep(wait_time)
#             else:
#                 logger.error(f"Embedding failed after {max_retries} attempts")
#                 raise

# def create_vectorstore_from_texts(texts):
#     """Create vector store with smart batch embedding"""
#     # Use smaller batch size for embedding
#     batch_size = 3  # Reduced batch size for Google API
#     all_embeddings = []
    
#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i + batch_size]
#         logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch_texts)} chunks)")
#         batch_embeddings = embed_with_retry(batch_texts)
#         all_embeddings.extend(batch_embeddings)
#         time.sleep(0.5)  # Brief pause between batches
    
#     # Create FAISS index from embeddings
#     return FAISS.from_embeddings(
#         text_embeddings=list(zip(texts, all_embeddings)),
#         embedding=embeddings_model
#     )

# def load_chain(vectordb):
#     llm = create_llm()
#     retriever = vectordb.as_retriever(search_kwargs={"k": 3})
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt_template}
#     )
#     return qa

# # Main processing
# if st.button("Get Answer"):
#     if uploaded_files and query:
#         with st.spinner("Processing..."):
#             all_vectorstores = []
            
#             for uploaded_file in uploaded_files:
#                 file_bytes = uploaded_file.read()
#                 file_hash = generate_pdf_hash(file_bytes)
#                 uploaded_file.seek(0)  # Reset file pointer
                
#                 db_path = VECTORSTORE_DIR / f"db_faiss_{file_hash}"
                
#                 # Load existing or create new vector store
#                 if db_path.exists():
#                     st.info(f"Using cached vector store for: {uploaded_file.name}")
#                     vectordb = FAISS.load_local(str(db_path), embeddings_model)
#                 else:
#                     with st.spinner(f"Processing {uploaded_file.name}..."):
#                         text = load_pdf_text(uploaded_file)
#                         if not text.strip():
#                             st.warning(f"No text extracted from {uploaded_file.name}. Skipping.")
#                             continue
                            
#                         docs = split_text(text)
#                         st.info(f"Split {uploaded_file.name} into {len(docs)} chunks")
                        
#                         try:
#                             vectordb = create_vectorstore_from_texts(docs)
#                             vectordb.save_local(str(db_path))
#                             st.success(f"Created vector store for: {uploaded_file.name}")
#                         except Exception as e:
#                             st.error(f"Failed to create vector store for {uploaded_file.name}: {str(e)}")
#                             continue
                
#                 all_vectorstores.append(vectordb)
            
#             if not all_vectorstores:
#                 st.error("No valid vector stores created. Please check your PDFs.")
#                 st.stop()
            
#             # Merge vector stores
#             if len(all_vectorstores) > 1:
#                 with st.spinner("Combining documents..."):
#                     combined_vectorstore = all_vectorstores[0]
#                     for store in all_vectorstores[1:]:
#                         combined_vectorstore.merge_from(store)
#             else:
#                 combined_vectorstore = all_vectorstores[0]
            
#             # Perform QA
#             with st.spinner("Generating answer..."):
#                 try:
#                     qa_chain = load_chain(combined_vectorstore)
#                     result = qa_chain({"query": query})
#                     st.subheader("Answer:")
#                     st.write(result["result"])
                    
#                     # Show sources
#                     with st.expander("Source Documents"):
#                         for i, doc in enumerate(result["source_documents"]):
#                             st.markdown(f"**Source {i+1}:**")
#                             st.caption(doc.page_content)
#                             st.write("---")
#                 except Exception as e:
#                     st.error(f"Error generating answer: {str(e)}")
#     else:
#         st.warning("Please upload at least one PDF and enter a question.")







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
import nest_asyncio
import logging
import traceback

nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
VECTORSTORE_DIR = Path("./vectorstore")
VECTORSTORE_DIR.mkdir(exist_ok=True)

# Smaller timeout to avoid long stalls
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
    request_options={'timeout': 60}
)

def create_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        google_api_key=GOOGLE_API_KEY,
        request_options={'timeout': 60}
    )

prompt_template = PromptTemplate.from_template("""
You are a helpful assistant answering based on the content of uploaded PDFs.

Context:
{context}

Question:
{question}

Answer:
""")

st.title("📄 PDF Question Answering with Gemini (Stable Embeddings)")

uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
query = st.text_input("Enter your question")

def generate_pdf_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def load_pdf_text(uploaded_file):
    text = ""
    uploaded_file.seek(0)
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # smaller chunks
        chunk_overlap=50,
        length_function=len
    )
    return splitter.split_text(text)

def create_vectorstore_from_texts(texts):
    all_embeddings = []
    for idx, chunk in enumerate(texts, start=1):
        st.info(f"Embedding chunk {idx}/{len(texts)}")
        try:
            emb = embeddings_model.embed_documents([chunk])
        except Exception as e:
            st.error(f"Embedding failed for chunk {idx}: {e}")
            st.stop()
        all_embeddings.extend(emb)
        time.sleep(0.2)
    return FAISS.from_embeddings(
        text_embeddings=list(zip(texts, all_embeddings)),
        embedding=embeddings_model
    )

def load_chain(vectordb):
    llm = create_llm()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

if st.button("Get Answer"):
    if uploaded_files and query:
        all_vectorstores = []

        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            file_hash = generate_pdf_hash(file_bytes)
            uploaded_file.seek(0)

            db_path = VECTORSTORE_DIR / f"db_faiss_{file_hash}"

            if db_path.exists():
                st.info(f"Loading cached vector store for {uploaded_file.name}")
                vectordb = FAISS.load_local(str(db_path), embeddings_model)
            else:
                st.info(f"Extracting text from {uploaded_file.name}...")
                text = load_pdf_text(uploaded_file)
                if not text.strip():
                    st.warning("No text extracted, skipping.")
                    continue

                st.info("Splitting text...")
                docs = split_text(text)

                st.info("Creating vector store...")
                vectordb = create_vectorstore_from_texts(docs)
                vectordb.save_local(str(db_path))
                st.success(f"Vector store created for {uploaded_file.name}")

            all_vectorstores.append(vectordb)

        if not all_vectorstores:
            st.error("No valid vector stores created.")
            st.stop()

        combined_vectorstore = all_vectorstores[0]
        for store in all_vectorstores[1:]:
            combined_vectorstore.merge_from(store)

        st.info("Generating answer...")
        try:
            qa_chain = load_chain(combined_vectorstore)
            result = qa_chain({"query": query})
            st.subheader("Answer:")
            st.write(result["result"])
            with st.expander("Sources"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.caption(doc.page_content)
        except Exception:
            st.error(f"Error during LLM call:\n{traceback.format_exc()}")

    else:
        st.warning("Please upload PDFs and enter a question.")


