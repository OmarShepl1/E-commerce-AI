# E-commerce Product QA Chatbot API

A Flask-based intelligent chatbot API that answers product-related questions using semantic search and large language models. It leverages FAISS for vector similarity, LangChain for context-aware QA, and TinyLlama for natural language response generation.

# Features

- Question answering based on a CSV dataset of product descriptions.
- Semantic similarity search using **SentenceTransformer** and **FAISS**.
- Context-aware answers using **LangChain** and **TinyLlama-1.1B-Chat**.
- RESTful API with JSON responses and CORS support.
- Preprocessing pipeline to tokenize, embed, and store product data.

# Project Structure

```
.
├── chat.py                   # Flask API server
├── products_dataset.csv      # Product dataset with descriptions
├── model_cache/              # Cache directory for Hugging Face models
└── requirements.txt          # Python dependencies
```

# Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ecommerce-chatbot-api.git
cd ecommerce-chatbot-api

# 2. Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

# Usage

```bash
# Run the Flask app
python chat.py
```

The app loads the dataset, builds the vector store, initializes the LLM, and starts the API.

# API Endpoints

- `POST /ask` – Accepts a JSON payload `{ "question": "..." }` and returns an answer and context.
- `POST /start_conversation` – Starts a new conversation session.
- `GET /` – Returns API status and usage instructions.

# Technologies Used

- Python, Flask
- Hugging Face Transformers & TinyLlama
- LangChain (QA Chain)
- FAISS (Vector Store)
- SentenceTransformer (Embeddings)
- pandas, torch
