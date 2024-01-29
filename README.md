# Betterzila_Gen_AI_Assignment
This repo is of a custom Chatbot over your documents using LangChain and Gradio. 

This also uses:

GPT4AllEmbeddings
ChromaDB for a vectorstore
From AnyScale Llama-2-70b-chat-hf for a text generation model
## Setup
To setup, please install requirements with pip install -r requirements.txt

Then, sign in to Anyscale and have credentials

ANYSCALE_API_BASE=...
ANYSCALE_API_KEY=....
ANYSCALE_MODEL_NAME=..

## Ingest
First, we need to ingest data. For this example, we are passing pdf of  "The 48 Laws Of Power". You can modify the code in ingest.py to ingest anything you want. To ingest, run python ingest.py
