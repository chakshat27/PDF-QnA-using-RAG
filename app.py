#RAG

# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplatea
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from pathlib import Path
# import hashlib
# import time
# import nest_asyncio
# import logging
# import traceback

# nest_asyncio.apply()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
# VECTORSTORE_DIR = Path("./vectorstore")
# VECTORSTORE_DIR.mkdir(exist_ok=True)

# # Smaller timeout to avoid long stalls
# embeddings_model = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=GOOGLE_API_KEY,
#     request_options={'timeout': 60}
# )

# def create_llm():
#     return ChatGoogleGenerativeAI(
#         model="models/gemini-2.5-pro",
#         google_api_key=GOOGLE_API_KEY,
#         request_options={'timeout': 60}
#     )

# prompt_template = PromptTemplate.from_template("""
# You are a helpful assistant answering based on the content of uploaded PDFs.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """)

# st.title("📄 PDF Q/A using RAG")

# uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
# query = st.text_input("Enter your question")

# def generate_pdf_hash(file_bytes):
#     return hashlib.md5(file_bytes).hexdigest()

# def load_pdf_text(uploaded_file):
#     uploaded_file.seek(0)
#     text = ""
#     pdf_reader = PdfReader(uploaded_file)
#     for page in pdf_reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text
#     return text

# def split_text(text):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len
#     )
#     return splitter.split_text(text)

# def create_vectorstore_from_texts(texts):
#     st.info("Embeddings...")
#     all_embeddings = []
#     for chunk in texts:
#         try:
#             emb = embeddings_model.embed_documents([chunk])
#         except Exception as e:
#             st.error(f"Embedding failed: {e}")
#             st.stop()
#         all_embeddings.extend(emb)
#         time.sleep(0.2)
#     return FAISS.from_embeddings(
#         text_embeddings=list(zip(texts, all_embeddings)),
#         embedding=embeddings_model
#     )

# def load_chain(vectordb):
#     llm = create_llm()
#     retriever = vectordb.as_retriever(search_kwargs={"k": 3})
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt_template}
#     )

# if st.button("Get Answer"):
#     if uploaded_files and query:
#         all_vectorstores = []

#         for uploaded_file in uploaded_files:
#             file_bytes = uploaded_file.read()
#             file_hash = generate_pdf_hash(file_bytes)
#             uploaded_file.seek(0)

#             db_path = VECTORSTORE_DIR / f"db_faiss_{file_hash}"

#             if db_path.exists():
#                 vectordb = FAISS.load_local(str(db_path), embeddings_model,
#                                             allow_dangerous_deserialization=True)
#             else:
#                 st.info("Extracting text...")
#                 text = load_pdf_text(uploaded_file)
#                 if not text.strip():
#                     st.warning("No text extracted, skipping.")
#                     continue

#                 st.info("Splitting text...")
#                 docs = split_text(text)

#                 st.info("Creating vector store...")
#                 vectordb = create_vectorstore_from_texts(docs)
#                 vectordb.save_local(str(db_path))
#                 st.success("Vector store created.")

#             all_vectorstores.append(vectordb)

#         if not all_vectorstores:
#             st.error("No valid vector stores created.")
#             st.stop()

#         combined_vectorstore = all_vectorstores[0]
#         for store in all_vectorstores[1:]:
#             combined_vectorstore.merge_from(store)

#         st.info("Generating answer...")
#         try:
#             qa_chain = load_chain(combined_vectorstore)
#             result = qa_chain({"query": query})
#             st.subheader("Answer:")
#             st.write(result["result"])
#             with st.expander("Sources"):
#                 for i, doc in enumerate(result["source_documents"]):
#                     st.markdown(f"**Source {i+1}:**")
#                     st.caption(doc.page_content)
#         except Exception:
#             st.error(f"Error during LLM call:\n{traceback.format_exc()}")

#     else:
#         st.warning("Please upload PDFs and enter a question.")






#RAG + LLM


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

st.title("📄 PDF Q/A")

uploaded_files = st.file_uploader("Upload one or more PDFs (optional)", type="pdf", accept_multiple_files=True)
query = st.text_input("Enter your question")

def generate_pdf_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def load_pdf_text(uploaded_file):
    uploaded_file.seek(0)
    text = ""
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return splitter.split_text(text)

def create_vectorstore_from_texts(texts):
    st.info("Embeddings...")
    all_embeddings = []
    for chunk in texts:
        try:
            emb = embeddings_model.embed_documents([chunk])
        except Exception as e:
            st.error(f"Embedding failed: {e}")
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

if st.button("Get Answer") and query:
    try:
        if uploaded_files:
            # Build or load vectorstores from PDFs
            all_vectorstores = []

            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                file_hash = generate_pdf_hash(file_bytes)
                uploaded_file.seek(0)

                db_path = VECTORSTORE_DIR / f"db_faiss_{file_hash}"

                if db_path.exists():
                    vectordb = FAISS.load_local(str(db_path), embeddings_model,
                                                allow_dangerous_deserialization=True)
                else:
                    st.info("Extracting text...")
                    text = load_pdf_text(uploaded_file)
                    if not text.strip():
                        st.warning("No text extracted, skipping.")
                        continue

                    st.info("Splitting text...")
                    docs = split_text(text)

                    st.info("Creating vector store...")
                    vectordb = create_vectorstore_from_texts(docs)
                    vectordb.save_local(str(db_path))
                    st.success("Vector store created.")

                all_vectorstores.append(vectordb)

            if not all_vectorstores:
                st.warning("No text found in PDFs. Switching to general LLM mode.")
                llm = create_llm()
                response = llm.invoke(query)
                answer = response.content if hasattr(response, "content") else str(response)
                st.subheader("Answer:")
                st.write(answer)
            else:
                combined_vectorstore = all_vectorstores[0]
                for store in all_vectorstores[1:]:
                    combined_vectorstore.merge_from(store)

                st.info("Generating answer from PDFs...")
                qa_chain = load_chain(combined_vectorstore)
                result = qa_chain({"query": query})

                if result["result"].strip():
                    st.subheader("Answer (from PDF):")
                    st.write(result["result"])
                    with st.expander("Sources"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.caption(doc.page_content)
                else:
                    st.info("No relevant content in PDFs. Switching to general LLM mode...")
                    llm = create_llm()
                    answer = llm.predict(query)
                    st.subheader("Answer (general):")
                    st.write(answer)

        else:
            # General LLM mode without PDFs
            st.info("No PDFs uploaded. Answering as general LLM...")
            llm = create_llm()
            answer = llm.predict(query)
            st.subheader("Answer:")
            st.write(answer)

    except Exception as e:
        st.error(f"Error:\n{traceback.format_exc()}")







