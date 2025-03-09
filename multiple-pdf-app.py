## Multiple PDF
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
import uuid

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

# Create a persistent directory for the vector store
CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

# Initialize session state variables
if 'store' not in st.session_state:
    st.session_state.store = {}
    
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

if 'chroma_initialized' not in st.session_state:
    st.session_state.chroma_initialized = False

if groq_api_key:
    llm = ChatGroq(api_key=groq_api_key, model_name="Gemma2-9b-It")
    
    ## Chat Interface
    session_id = st.text_input("Session ID", value="default_session")
        
    uploaded_files = st.file_uploader("Choose a PDF File", type="pdf", 
                                      accept_multiple_files=True)

    # Process new uploaded files
    if uploaded_files:
        # Create or get the vector store
        if st.session_state.chroma_initialized:
            chroma = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        else:
            chroma = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            st.session_state.chroma_initialized = True
        
        # Process only new files
        new_docs = []
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uuid.uuid4()}"
            
            if file_id not in st.session_state.processed_files:
                st.session_state.processed_files.add(file_id)
                
                temp_pdf = f"{uploaded_file.name}"
                with open(temp_pdf, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    loader = PyPDFLoader(temp_pdf)
                    file_docs = loader.load()
                    if file_docs:
                        new_docs.extend(file_docs)
                        st.success(f"Successfully processed: {uploaded_file.name}")
                    else:
                        st.warning(f"No content extracted from: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # If there are new documents to add to the vector store
        if new_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, 
                                                          chunk_overlap=750)
            splits = text_splitter.split_documents(new_docs)
            
            # Add documents to existing collection
            chroma.add_documents(documents=splits)
            st.success(f"Added {len(splits)} chunks to the vector store")
            
            # Display total documents count
            st.info(f"Total documents in vector store: {len(chroma.get()['documents'])}")
        
        # Create retriever from the updated vector store
        retriever = chroma.as_retriever()
        retriever.search_kwargs["k"] = 3
        
        contextualize_QA_system_prompt = (
            "Given a chat history and the latest user question "
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
            "You are an AI assistant that helps people find information for Q&A tasks. "
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
        
        # Store the chain in session state so it can be accessed outside this block
        st.session_state.conversational_rag_chain = conversational_rag_chain
        st.session_state.session_id = session_id
    
    # If there are any files processed, show the query interface
    if 'processed_files' in st.session_state and st.session_state.processed_files:
        user_input = st.text_input("Your question:")
        
        if user_input and 'conversational_rag_chain' in st.session_state:
            with st.spinner("Thinking..."):
                session_history = get_session_history(st.session_state.session_id)
                response = st.session_state.conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    },
                )
                st.write("Assistant:", response['answer'])
                
                # Show chat history in a cleaner format
                st.subheader("Chat History")
                for i, msg in enumerate(session_history.messages):
                    if msg.type == "human":
                        st.markdown(f"**You:** {msg.content}")
                    else:
                        st.markdown(f"**Assistant:** {msg.content}")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF file to start chatting")
else:
    st.warning("Please enter the Groq API Key")