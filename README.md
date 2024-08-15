# Streamlit RAG Bot

A Retrieval-Augmented Generation (RAG) chatbot with memory, built using Streamlit, open-source embeddings, and PDF processing capabilities.

![Streamlit RAG Bot Demo](path_to_your_demo_image.gif)

## Demo

Try out the live demo: [Streamlit RAG Bot Demo](http://rag-chat-app.streamlit.app/)

## Description

This project implements a conversational AI system that can read and understand PDF documents, then answer questions about their content. It uses a RAG (Retrieval-Augmented Generation) approach, combining the power of large language models with a retrieval system that fetches relevant information from the uploaded PDF.

Key features:
- PDF upload and processing
- Conversation memory
- Open-source embeddings for efficient text representation
- Streamlit-based user interface for easy interaction

![PDF Upload Interface](path_to_pdf_upload_image.png)

## Technologies Used

- Streamlit: For creating the web application interface
- LangChain: For building the conversational AI pipeline
- FAISS: For efficient similarity search and clustering of dense vectors
- HuggingFace Transformers: For open-source embeddings
- PyPDF2 and langchain_community.document_loaders: For PDF processing
- Groq: As the large language model for text generation

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/streamlit-rag-bot.git
   cd streamlit-rag-bot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run streamlit_rag_app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Upload a PDF file using the file uploader.

4. Once the PDF is processed, you can start asking questions about its content in the chat interface.

![Chat Interface](path_to_chat_interface_image.png)

## Features

- PDF Upload: Users can upload PDF files which are then processed and indexed.
- Conversational Interface: Chat-like interface for asking questions about the PDF content.
- Context-Aware Responses: The system uses the chat history to provide more accurate and contextual answers.
- Efficient Retrieval: Utilizes FAISS for fast and efficient similarity search in the vector space.
- Open-Source Embeddings: Uses HuggingFace's GTE-base model for generating text embeddings.

## Project Structure

```
streamlit-rag-bot/
│
├── streamlit_rag_app.py   # Main Streamlit application file
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not tracked in git)
├── .rag_with_memory       # backend code (For understanding)
├── .rag_without_memory    # For understanding
└── README.md              # Project documentation
```

## How It Works

1. **PDF Processing**: When a user uploads a PDF, the system extracts text from it using PyPDF2 and splits it into manageable chunks.

2. **Embedding Generation**: The text chunks are converted into vector embeddings using the HuggingFace GTE-base model.

3. **Vector Storage**: These embeddings are stored in a FAISS index for efficient retrieval.

4. **User Interaction**: Users can ask questions through the Streamlit chat interface.

5. **Context-Aware Retrieval**: The system uses the chat history and the current question to formulate a context-aware query.

6. **Answer Generation**: Relevant text chunks are retrieved from the FAISS index and fed into the Groq language model to generate a response.

7. **Conversation Memory**: The chat history is maintained to provide context for future questions.

## Contextualizing Questions

One of the key features of this RAG bot is its ability to contextualize questions based on the conversation history. This is achieved through the following process:

1. **Chat History Analysis**: The system examines the previous interactions in the chat.

2. **Question Reformulation**: If the current question references context from earlier in the conversation, the system reformulates it into a standalone question.

3. **Context Integration**: The reformulated question incorporates relevant context from the chat history.

4. **Improved Retrieval**: This contextualized question leads to more accurate retrieval of relevant information from the PDF.

Example:
- User: "What is the main topic of the document?"
- Bot: "The main topic is climate change."
- User: "What does it say about its effects?"
- Bot: (Internally contextualizes to "What does the document say about the effects of climate change?") "The document discusses several effects of climate change, including rising sea levels, increased frequency of extreme weather events, and impacts on biodiversity."

This contextual awareness allows for more natural and coherent conversations about the PDF content.

## Configuration

The application uses several configurable parameters:

- `chunk_size`: The size of text chunks for processing (default: 1000)
- `chunk_overlap`: The overlap between text chunks (default: 200)
- `model_name`: The name of the HuggingFace model for embeddings (default: "thenlper/gte-base")
- `llm_model_name`: The name of the LLM model (default: "llama-3.1-8b-instant")

These can be adjusted in the `streamlit_rag_app.py` file to optimize performance for different use cases.

## Limitations

- The system is designed for moderate-sized PDFs. Very large documents may require additional optimization.
- The quality of answers depends on the content of the PDF and the capabilities of the underlying language model.
- Internet connectivity is required for using the Groq API and downloading the embedding model.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Groq](https://groq.com/)

## Contact

Mariya Abishek S - mariyaabishek86@gamil.com

Project Link: [https://github.com/Abishek318/streamlit-rag-bot](https://github.com/Abishek318/streamlit-rag-bot)
