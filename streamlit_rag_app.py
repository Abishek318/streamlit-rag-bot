import streamlit as st
import tempfile
import os
import uuid
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import PyPDF2
import logging 

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

load_dotenv()

# Create model object
@st.cache_resource
def get_llm():
    try:
        return ChatGroq(model="llama-3.1-8b-instant")
    except Exception as e:
        st.error(f"Error initializing ChatGroq: {str(e)}")
        return None

# Pull embedding model from HuggingFace
@st.cache_resource
def get_embeddings():
    try:
        model_name = "thenlper/gte-base"
        model_kwargs = {'device': 'cpu'}
        # model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': False}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        st.error(f"Error initializing HuggingFaceEmbeddings: {str(e)}")
        return None

def create_conversational_rag_chain(retriever):
    llm = get_llm()
    if not llm:
        return None
    
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Statefully manage chat history ###
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain, store

# @st.cache_resource
def extract_text_from_pdf(pdf_file,session_id):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        # Check if the PDF is readable
        try:
            with open(tmp_file_path, 'rb') as f:
                PyPDF2.PdfReader(f)
        except PyPDF2.errors.PdfReadError:
            raise ValueError("The uploaded PDF file is not readable or is corrupted.")

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No text could be extracted from the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = get_embeddings()
        if not embeddings:
            raise ValueError("Failed to initialize embeddings.")
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        os.unlink(tmp_file_path)  # Delete the temporary file
        return retriever
    except Exception as e:
        logger.error(e,exc_info=True)
        st.error(f"Error processing PDF: {str(e)}")
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)  # Ensure temporary file is deleted even if an error occurs
        return None

def main_screen():
    st.title("📚 PDF Chat App")
    st.write("Welcome! Please upload your PDF file to begin.")

    uploaded_file = st.file_uploader("Select a PDF file with minimal size for a faster response.", type="pdf")
    
    if uploaded_file is not None:
        # if uploaded_file.size <= 2000000:  # 2MB = 2,000,000 bytes
            with st.spinner("Processing PDF..."):
                session_id=str(uuid.uuid4())
                retriever = extract_text_from_pdf(uploaded_file,session_id)
                if retriever:
                    st.session_state.retriever = retriever
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.conversational_rag_chain, st.session_state.history_store = create_conversational_rag_chain(retriever)
                    if st.session_state.conversational_rag_chain:
                        st.success("PDF uploaded and processed successfully!")
                        st.session_state.page = "chat"
                        st.session_state.session_id = session_id
                        st.rerun()
                    else:
                        st.error("Failed to create conversation chain. Please try again.")
                else:
                    st.error("Failed to process the PDF. Please try another file.")
        # else:
        #     st.error("File size exceeds 2MB. Please upload a smaller file.")

def chat_screen():
    st.title("💬 Talk to your PDF")
    
    st.sidebar.title("PDF Info")
    st.sidebar.info(f"PDF: {st.session_state.pdf_name}")
    if st.sidebar.button("Upload New PDF"):
        st.session_state.page = "main"
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.session_state.conversational_rag_chain = None
        st.session_state.history_store = None
        st.session_state.session_id = None
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input and st.session_state.conversational_rag_chain:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in st.session_state.conversational_rag_chain.stream(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    },
                ):
                    if isinstance(chunk, dict):
                        content = chunk.get('answer') or chunk.get('text') or chunk.get('content') or ''
                        if content:
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                    elif isinstance(chunk, str):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                
                if full_response:
                    response_placeholder.markdown(full_response)
                else:
                    response_placeholder.markdown("I'm sorry, I couldn't generate a response.")
            except Exception as e:
                st.error(f"An error occurred while generating the response: {str(e)}")
                full_response = "I encountered an error while trying to respond. Please try again."
                response_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.session_id in st.session_state.history_store:
            del st.session_state.history_store[st.session_state.session_id]
        st.rerun()

def main():
    st.set_page_config(page_title="PDF Chat App", page_icon="📚", layout="wide")

    # Initialize session state variables
    if "page" not in st.session_state:
        st.session_state.page = "main"
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = ""
    if "conversational_rag_chain" not in st.session_state:
        st.session_state.conversational_rag_chain = None
    if "history_store" not in st.session_state:
        st.session_state.history_store = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    if st.session_state.page == "main":
        main_screen()
    elif st.session_state.page == "chat":
        chat_screen()

if __name__ == "__main__":
    main()