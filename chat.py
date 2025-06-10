from flask import Flask, request, jsonify, session, render_template_string, redirect
import pandas as pd
import torch
import os
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer
import traceback
import secrets
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

os.makedirs("cached_model", exist_ok=True)
CACHE_DIR = "model_cache"


vector_db = None
stuff_chain = None


def initialize_app():
    global vector_db, stuff_chain
    
    print("Starting initialization...")
    
    
    try:
        print("Loading dataset...")
        df = pd.read_csv("products_dataset (1).csv")
        print(f"Dataset loaded with {len(df)} products")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    
    print("Preparing documents...")
    documents = df["description"].astype(str).tolist()
    metadatas = [
        {
            "product_id": row["product_id"],
            "title": row["title"]
        }
        for _, row in df.iterrows()
    ]
    
    
    print("Splitting documents...")
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.create_documents(documents, metadatas=metadatas)
    
    
    print("Generating embeddings...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, convert_to_tensor=False)
    
    
    print("Creating vector database...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = FAISS.from_documents(chunks, embedding_model)
    
    
    print("Loading language model...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir=CACHE_DIR)
    model1 = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    
    
    print("Creating text generation pipeline...")
    pipe = pipeline(
        "text-generation",
        model=model1, 
        tokenizer=tokenizer,
        max_length=512
    )
    
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    
    print("Setting up QA chain...")
    qna_template = "\n".join([
        "You are an E-commerce chatbot assistant. Answer the next question using the provided context.",
        "If the answer is not contained in the context, say 'NO ANSWER IS AVAILABLE'.",
        "### Context:",
        "{context}",
        "",
        "### Question:",
        "{question}",
        "",
        "### Answer:",
    ])
    
    qna_prompt = PromptTemplate(
        template=qna_template,
        input_variables=['context', 'question']
    )
    
    stuff_chain = load_qa_chain(llm, chain_type="stuff", prompt=qna_prompt)
    
    print("Initialization complete!")
    return True

def generate_secret_key():
    return secrets.token_hex(24)

app.secret_key = generate_secret_key()

@app.route('/ask', methods=['POST'])
def ask():
    try:
        
        if request.is_json:
            question = request.json.get('question')
        else:
            question = request.form.get('question')
        
        
        if not question:
            return jsonify({
                'answer': "Error: No question provided",
                'context': [],
                'conversation_id': session.get('conversation_id', "new_session")
            }), 400
        
        
        if 'conversation_id' not in session:
            session['conversation_id'] = "new_session"
        
        
        if vector_db is None or stuff_chain is None:
            return jsonify({
                'answer': "Error: Models not initialized. Please wait or restart the server.",
                'context': [],
                'conversation_id': session.get('conversation_id')
            }), 500
            
        
        similar_docs = vector_db.similarity_search(question, k=3)
        response = stuff_chain({"input_documents": similar_docs, "question": question}, return_only_outputs=True)
        
        
        output_text = response.get('output_text', 'No answer found')
        if '### Answer:' in output_text:
            answer_parts = output_text.split('### Answer:')
            if len(answer_parts) > 1 and answer_parts[1].strip():
                answer = answer_parts[1].strip()
            else:
                answer = "No clear answer found in the context"
        else:
            answer = output_text

        return jsonify({
            'answer': answer,
            'context': [doc.page_content for doc in similar_docs],
            'conversation_id': session.get('conversation_id')
        })
    except Exception as e:
        
        traceback.print_exc()
        return jsonify({
            'answer': f"Server error: {str(e)}",
            'context': [],
            'conversation_id': session.get('conversation_id', "new_session")
        }), 500

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    try:
        
        session['conversation_id'] = secrets.token_hex(16)
        return jsonify({
            'message': 'New conversation started. Feel free to ask any questions!',
            'conversation_id': session['conversation_id']
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'message': f"Server error: {str(e)}"
        }), 500

@app.route('/', methods=['GET'])
def index():
    global vector_db, stuff_chain
    status = "ready" if vector_db is not None and stuff_chain is not None else "initializing"
    return jsonify({
        "instructions": "Send POST requests to /ask with a 'question' field",
        "message": "E-commerce Chatbot API is running",
        "status": status
    })

if __name__ == '__main__':
    print("Starting E-commerce Chatbot Server...")
    
    initialization_successful = initialize_app()
    
    if initialization_successful:
        print("Initialization successful. Starting server...")
        app.run(debug=True)
    else:
        print("Failed to initialize application. Check the logs for details.")