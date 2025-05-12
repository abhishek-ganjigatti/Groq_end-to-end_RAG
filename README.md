# 🧠 Retrieval-Augmented Generation (RAG) with LangChain, FAISS, and Groq

This project is a complete implementation of a **Retrieval-Augmented Generation (RAG)** pipeline. It combines document retrieval with large language model inference to answer user queries with grounded, contextual information.

## 🚀 Features

* Load documents from the web (LangChain documentation site)
* Split text into retrievable chunks
* Generate embeddings using **Ollama**
* Store embeddings in a **FAISS** vector store
* Retrieve relevant chunks based on user queries
* Generate accurate responses using **Groq’s Chat model**
* Interactive interface built with **Streamlit**

---

## 🧱 Components Overview

### 1. Document Loading

Documents are fetched from a web URL using:

```python
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.langchain.com/")
docs = loader.load()
```

### 2. Text Splitting

Splitting the documents into manageable chunks:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
```

### 3. Embedding Generation

Converting text into vector representations:

```python
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings()
```

### 4. Vector Store with FAISS

Storing and indexing embeddings for fast similarity search:

```python
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()
```

### 5. RAG Chain Construction

Creating a retrieval chain to process queries:

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatGroq

llm = ChatGroq()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

### 6. Prompt Construction

The model is instructed to only use the retrieved context via a structured prompt template.

### 7. Inference and UI

Streamlit UI allows users to enter a question, get a response, and view the source context:

```python
import streamlit as st
query = st.text_input("Ask a question:")
if query:
    result = qa_chain.run(query)
    st.write("### Answer:", result)
```

---

## 🖼️ UI Example

A clean and minimal Streamlit interface to:

* Input queries
* Display model answers
* Optionally show source context

---

## 🛠️ Requirements

* Python 3.9+
* `langchain`
* `faiss-cpu`
* `streamlit`
* `ollama`
* `groq` SDK (or appropriate integration for `ChatGroq`)

---

## ✅ To Run

1. Clone the repo.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit:

   ```bash
   streamlit run app.py
   ```

---

## 📝 Notes

* Ensure `Ollama` and `Groq` API keys or backends are configured correctly.
* Minor typo fixed: **"Docuemt" → "Document" Similarity Search**.
* You can improve formatting or truncate long documents in the UI for better readability.

---

## 📚 Summary

This project showcases an **end-to-end RAG implementation** using LangChain for retrieval, FAISS for indexing, and Groq for generative responses. It provides a scalable way to build intelligent, document-aware AI applications.

---

#### Acknowledgements

I learned about this implementation from [Krish Naik](https://github.com/krishnaik06) and his excellent [Langchain with Groq repository](https://github.com/krishnaik06/Updated-Langchain/tree/main/groq).
####
