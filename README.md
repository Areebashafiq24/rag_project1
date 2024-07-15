# Retrieval-Augmented Generation (RAG) Implementation

## Description
This project implements a Retrieval-Augmented Generation (RAG) system using the SQuAD dataset. 
It cleans and splits the text data into chunks, embeds the processed documents using Sentence Transformers, 
and indexes them with FAISS. A GPT-Neo model is then used to generate responses based on retrieved document chunks. 
The setup includes loading and processing the dataset, embedding and indexing the documents, and finally generating 
responses to user queries.

## Setup

### Prerequisites

- Python 3.12.4
- Virtual Environment (optional but recommended)


### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/rag-project.git
   cd rag-project



### Usage
# Loading the Dataset

from datasets import load_dataset
ds = load_dataset("rajpurkar/squad")




### processing the dataset
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


def clean_markdown(text):

    """Clean Markdown syntax from text."""
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'!\[[^\]]]*]\([^)]*\)', '', text)
    text = re.sub(r'#+\s?', '', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'-{2,}', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()  # Strip extra spaces

def process_context(entry, chunk_size, chunk_overlap):

    """Process a single context and return document chunks."""
    
    context = entry['context']
    clean_context = clean_markdown(context)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(clean_context)
    processed_docs = []
    for j, chunk in enumerate(docs):
        metadata = {
            "Document ID": entry['id'],
            "Chunk Number": j + 1,
        }
        header = f"Document ID: {entry['id']}\n"
        for key, value in metadata.items():
            header += f"{key}: {value}\n"
        chunk_with_header = header + chunk
        processed_docs.append(chunk_with_header)

    return processed_docs

# Parameters for text splitting
    chunk_size = 1200
    chunk_overlap = 100
    processed_docs = []
    for entry in ds['train'].select(range(50000)):
       processed_docs.extend(process_context(entry, chunk_size, chunk_overlap))

# EMBEDDING AND INDEXING 
    from sentence_transformers import SentenceTransformer
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings

# Initialize the model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    hf_embedding = HuggingFaceEmbeddings(model_name=model_name)

# Embed and index all the documents using FAISS
   db = FAISS.from_texts(processed_docs, hf_embedding)

# Save the indexed data locally
   db.save_local("faiss_AiDoc")



# Loading the FAISS Index and Metadata

    import faiss
    import numpy as np
    import pickle
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from sentence_transformers import SentenceTransformer

# Load FAISS index
   index = faiss.read_index("faiss_index.bin")

# Load metadata
     with open("faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
     processed_docs = metadata["processed_docs"]
     model_name = metadata["model_name"]

# Re-initialize the embedding model
     embedding_model = HuggingFaceEmbeddings (model_name=model_name)
     
     sentence_model = SentenceTransformer(model_name)

# Create the FAISS vector store
    faiss_index = FAISS(embedding_function=embedding_model, index=index, docstore=None, index_to_docstore_id=None)



# GENERATING RESPONSES 
     import torch
     from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load the GPT-Neo model and tokenizer
     model_name = "EleutherAI/gpt-neo-1.3B"
     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
     model = GPTNeoForCausalLM.from_pretrained(model_name)

# Set the pad token to be the eos token
     tokenizer.pad_token = tokenizer.eos_token
     def generate_response(query, retrieved_docs, max_input_length=1024, max_new_tokens=512):

     
# Combine the query and retrieved documents
    input_text = f"Query: {query}\nDocuments:\n{retrieved_docs}\nAnswer:"

# Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)

# Generate the response
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
   query = "What is the capital of France?"
# Combine the chunks of the top retrieved document as a single string
    retrieved_docs = " ".join([chunk for chunk in processed_docs[:5]])

    response = generate_response(query, retrieved_docs)
    print("Response:", response)






