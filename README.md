# 🧠 Conversational RAG with Chat History

A **Conversational Retrieval-Augmented Generation (RAG) pipeline** that enhances chatbot responses by retrieving relevant document context while maintaining conversation history.

## ✨ Features
✅ **Conversational Memory** – Maintains chat history for better context.  
✅ **RAG-powered Retrieval** – Retrieves top 🔝 3 relevant document chunks.  
✅ **PDF Handling** – Loads and processes documents efficiently.  
✅ **Standalone Question Reformulation** – Enhances user queries before retrieval.  
✅ **Configurable LLM Response** – Customizes chatbot answers based on prompts.  

---

## 🛠️ Tech Stack
- **LLM Provider**: Groq API  
- **Vector Store**: ChromaDB  
- **Frontend**: Streamlit  
- **Embedding Model**: Sentence Transformers  
- **Orchestration**: LangChain  

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```
### 2️⃣ Set Up API Key
Ensure you have a valid Groq API key. Create a .env file and add:
```env
GROQ_API_KEY="your_api_key_here"
```
Note: The .env file is not uploaded to GitHub (.gitignore includes it). Users must configure this themselves.
### 3️⃣ Run the App
```bash
streamlit run app.py
```
Note: 'app.py' only process one PDF, where 'multiple-pdf-app.py' process multiple PDFs.
## Project Workflow
### 📂 Document Processing
1️⃣ Upload PDFs

2️⃣ Save to Temporary Storage (temp_pdf/)

3️⃣ Load, Split & Embed Documents

4️⃣ Store in ChromaDB for Fast Retrieval

### 🔍 Retrieval-Augmented Generation
1️⃣ Retrieve Top k=3 Most Relevant Chunks

2️⃣ Refine User Input into a Standalone Question

3️⃣ Pass Reformulated Query to Retriever & LLM

4️⃣ Generate an Answer Based on Retrieved Context

### 🗣️ Conversational Memory
1️⃣ Stores chat history using RunnableWithMessageHistory

2️⃣ Session-based retrieval to maintain conversation flow

3️⃣ Improves chatbot’s ability to understand multi-turn interactions

## ⚙️ Configuration & Customization
### 🎯 Adjust Number of Retrieved Chunks
Modify k value to retrieve more/less context:
```python
retriever.search_kwargs["k"] = 3
```
### 📝 Customize the QA Prompt
Modify qa_prompt to change response style.

### 🔄 Enable Session-based Memory
Chat history is maintained using:
```python
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history(session_id),
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)
```
## 🏗️ Project Setup Notes
The .gitignore file ensures that the venv (virtual environment) and .env file are not uploaded to GitHub.

Users need to set up their own virtual environment and configure the API key before running the app.
## 🏆 Future Improvements
🔹 Fine-tune the LLM for better domain-specific responses

🔹 Improve retrieval with hybrid search (BM25 + Embeddings)

🔹 Optimize query reformulation using advanced LLM prompts
## 🤝 Contributing
Got an idea? Feel free to open an issue or submit a PR! 🚀

## 📜 License
This project is licensed under the MIT License.
