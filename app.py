## Single PDF
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Code below solves the error:
# Error from using ChromaDB - ValueError: Could not connect to 
# tenant default_tenant. Are you sure it exists? #26884
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()


from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Streamlit Setup
st.title("Conversational RAG with PDF uploads and Chat History")
st.write("Upload PDF's and chat with their content")

## Input the Groq API Key
groq_api_key = st.text_input("Enter your Groq API Key", type="password")

if groq_api_key:
    llm = ChatGroq(api_key=groq_api_key, model_name="Gemma2-9b-It")
    
    ## Chat Interface
    session_id = st.text_input("Session ID", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}
        
    uploaded_files = st.file_uploader("Choose a PDF File", type="pdf", 
                                      accept_multiple_files=True)

    if uploaded_files:
        docs = []
        for uploaded_file in uploaded_files:
            temp_pdf = f"{uploaded_file.name}"
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                
            loader = PyPDFLoader(temp_pdf)
            docs.extend(loader.load())

        ## Split and Embed the PDFs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, 
                                                       chunk_overlap=750)
        splits = text_splitter.split_documents(docs)
        chroma = Chroma.from_documents(documents=splits, 
                                       embedding=embeddings)
        retriever = chroma.as_retriever()
        retriever.search_kwargs["k"] = 3
        
        contextualize_QA_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_QA_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_QA_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, 
                                                                 retriever,
                                                                 contextualize_QA_prompt)
        
        ## Answer Question
        system_prompt = (
            "You are an AI assistant that helps people find information for Q&A tasks."
            "You are given the following extracted parts of a long document "
            "and a question. Provide a conversational answer based on the "
            "context provided. If the question cannot be answered using the "
            "information provided, say \"I don't know\". Don't try to make up an "
            "answer."
            "\n\nCONTEXT:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
        
        # Define the get_session_history function properly
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        # Pass the function directly, not its result
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, 
            get_session_history,  # Pass the function itself
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        user_input = st.text_input("Your question:")
        
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },
            )
            # st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the Groq API Key")