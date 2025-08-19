
# PDF + General Q/A Application

This Streamlit application allows users to ask questions from uploaded PDFs using **Retrieval-Augmented Generation (RAG)**. If no PDFs are uploaded or no relevant content is found, the app falls back to answering questions using **Google Gemini LLM**.

---

## PDF Query Application Architecture

The architecture consists of the following key components:

1. **Frontend (Streamlit)**  
   Provides a user-friendly interface for uploading PDF files, entering queries, and viewing answers.

2. **Backend**  
   Handles requests from the frontend, extracts text from PDFs, splits it into chunks, creates embeddings, stores them in a vector database, and generates answers using LLMs.

3. **AI Models**  
   - **LLM (ChatGoogleGenerativeAI)**: Generates context-aware answers from retrieved PDF content or directly from user queries.  
   - **Vector Embeddings**: PDF text is embedded using **GoogleGenerativeAIEmbeddings**. FAISS is used to store and efficiently retrieve these embeddings.

4. **Database (FAISS)**  
   Stores embeddings for fast retrieval of relevant content from uploaded PDFs. Vector stores are cached locally for efficiency.

---

## Dependencies

Ensure you have the following installed:

- Python 3.7+
- Streamlit
- PyPDF2
- LangChain
- FAISS
- Google Generative AI
- nest_asyncio

Install dependencies with:

```bash
pip install streamlit PyPDF2 langchain faiss-cpu google-generativeai nest_asyncio
````

---

## Setup

1. **Clone the Repository**:

```bash
git clone https://github.com/yourusername/pdf-qa-rag.git
cd pdf-qa-rag
```

2. **Set Google API Key in Streamlit Secrets**:

Create `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_google_api_key_here"
```

3. **Create Necessary Directories**:

```text
./vectorstore/      # Stores FAISS vector databases
```

---

## Running the Application

Run the app using:

```bash
streamlit run app.py
```

1. Upload one or more PDFs (optional).
2. Enter your question in the text input field.
3. Click **Get Answer**.

   * If PDFs contain relevant content, the app retrieves the answer from the vector store.
   * Otherwise, the app queries the LLM directly.

---

## Code Overview

### Importing Libraries

```python
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
```

---

### Core Functions

1. **PDF Hashing** – Generate unique hash for caching:

```python
def generate_pdf_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()
```

2. **Load PDF Text**:

```python
def load_pdf_text(uploaded_file):
    uploaded_file.seek(0)
    text = ""
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text
```

3. **Text Splitting**:

```python
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)
```

4. **Create Vector Store**:

```python
def create_vectorstore_from_texts(texts):
    all_embeddings = [embeddings_model.embed_documents([chunk]) for chunk in texts]
    return FAISS.from_embeddings(list(zip(texts, all_embeddings)), embedding=embeddings_model)
```

5. **Load Retrieval Chain**:

```python
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
```

---

### Streamlit UI

* Upload PDFs (optional)
* Enter a query
* Click **Get Answer**
* Answers are displayed along with PDF sources if relevant

---

## Notes

* FAISS vector stores are cached to avoid recomputing embeddings.
* Small delays (`time.sleep(0.2)`) prevent API rate limit issues.
* `nest_asyncio` ensures async calls work smoothly in Streamlit.

---
''''


