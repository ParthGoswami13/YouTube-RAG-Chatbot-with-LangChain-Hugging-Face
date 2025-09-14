# YouTube RAG Chatbot with LangChain & Hugging Face

A Retrieval-Augmented Generation (RAG) chatbot that allows you to ask questions about YouTube video content using LangChain and Hugging Face models.

## Overview

This project creates an intelligent chatbot that can answer questions about YouTube videos by:
- Extracting transcripts from YouTube videos
- Creating embeddings using Hugging Face models
- Storing the embeddings in a vector database
- Using retrieval-augmented generation to provide accurate answers based on video content

## Features

- **YouTube Video Processing**: Automatically extract transcripts from YouTube videos
- **Smart Q&A**: Ask questions about video content and get contextually relevant answers
- **Vector Search**: Efficient similarity search using embeddings
- **Open Source Models**: Uses Hugging Face transformers for embeddings and text generation
- **Interactive Interface**: User-friendly chatbot interface

## Technologies Used

- **LangChain**: Framework for building LLM applications
- **Hugging Face Transformers**: Open-source models for embeddings and text generation
- **YouTube Transcript API**: Extract video transcripts
- **Vector Database**: Store and retrieve document embeddings
- **Python**: Core programming language

## Prerequisites

- Python 3.8+
- YouTube videos with available transcripts
- Internet connection for downloading models and accessing YouTube

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ParthGoswami13/YouTube-RAG-Chatbot-with-LangChain-Hugging-Face.git
   cd YouTube-RAG-Chatbot-with-LangChain-Hugging-Face
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional dependencies**
   ```bash
   pip install langchain
   pip install transformers
   pip install youtube-transcript-api
   pip install sentence-transformers
   pip install faiss-cpu  # or faiss-gpu for GPU support
   pip install streamlit  # if using Streamlit interface
   ```

## Required Libraries

```python
# Core libraries
langchain
transformers
torch
youtube-transcript-api
sentence-transformers

# Vector database
faiss-cpu  # or faiss-gpu
chromadb   # alternative vector store

# Interface (optional)
streamlit
gradio

# Utility
pandas
numpy
```

## Usage

### Basic Usage

1. **Run the Jupyter notebook**
   ```bash
   jupyter notebook Youtube_Chatbot_rag_using_langchain.ipynb
   ```

2. **Provide YouTube Video ID**
   - Extract the video ID from the YouTube URL (e.g., from `https://www.youtube.com/watch?v=dQw4w9WgXcQ` use `dQw4w9WgXcQ`)
   - Enter the video ID when prompted
   - The system will automatically extract the transcript using the video ID

3. **Ask Questions**
   - Type your questions about the video content
   - Get AI-powered answers based on the video transcript

### Example Questions

- "What are the main topics discussed in this video?"
- "Can you summarize the key points about [specific topic]?"
- "What does the speaker say about [specific concept]?"
- "What are the recommendations mentioned in the video?"

## Configuration

### Model Selection

You can customize the models used:

```python
# Embedding model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Text generation model
llm_model = "microsoft/DialoGPT-medium"
```

### Vector Store Options

Choose your preferred vector database:
- **FAISS**: Fast similarity search
- **Chroma**: Persistent storage
- **Pinecone**: Cloud-based (requires API key)

## Architecture

```
YouTube URL → Transcript Extraction → Text Chunking → Embeddings → Vector Store
                                                                        ↓
User Question → Question Embedding → Similarity Search → Context Retrieval → LLM → Answer
```

## How It Works

1. **Video Processing**: Extract transcript from YouTube video using YouTube Transcript API
2. **Text Chunking**: Split transcript into manageable chunks for processing
3. **Embedding Creation**: Generate vector embeddings using Hugging Face models
4. **Vector Storage**: Store embeddings in a vector database for efficient retrieval
5. **Question Processing**: Convert user questions into embeddings
6. **Context Retrieval**: Find relevant transcript chunks using similarity search
7. **Answer Generation**: Use retrieved context to generate accurate answers

## Author

**Parth Goswami**
- GitHub: [@ParthGoswami13](https://github.com/ParthGoswami13)

## Acknowledgments

- LangChain team for the amazing framework
- Hugging Face for open-source models
- YouTube Transcript API contributors
- Open-source community

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Models](https://huggingface.co/models)
- [RAG Implementation Guide](https://python.langchain.com/docs/tutorials/rag/)
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)

## Troubleshooting

### Common Issues

**Transcript Not Available**
- Ensure the video has subtitles/captions enabled
- Try videos in English or your target language

**Memory Issues**
- Reduce chunk size for long videos
- Use CPU instead of GPU if facing memory constraints

**Model Loading Errors**
- Check internet connection for model downloads
- Verify Hugging Face model names and availability

---

If you find this project helpful, please give it a star on GitHub!
