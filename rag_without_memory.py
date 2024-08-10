from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import chromadb
import os

# set  GROQ_API_KEY

load_dotenv()

# Create model object
llm = ChatGroq(model="llama3-8b-8192")

# Pull embedding model from HuggingFace
model_name = "thenlper/gte-base"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

loader = PyPDFLoader("test_pdf\Stock market today.pdf")
docs = loader.load()


collection_name = "nifty50"
chromadb_directory = "chroma_db"
persist_directory = os.path.join(os.getcwd(), chromadb_directory)

persistent_client = chromadb.PersistentClient(path=persist_directory)


try:
    persistent_client.get_collection(collection_name)
    collection_exists=True
except ValueError :
    collection_exists=False

if collection_exists:
    print("Collection already exists")
    vectorstore = Chroma(collection_name=collection_name, embedding_function=hf_embeddings, persist_directory=persist_directory)
else:
    print("Collection does not exist")
    collection = persistent_client.create_collection(collection_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=hf_embeddings, persist_directory=persist_directory, collection_name=collection_name)

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5,"k":3})
# retriever.invoke("cricket")

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

print("Bot: " + rag_chain.invoke("Why did the Nifty market crash?"))
