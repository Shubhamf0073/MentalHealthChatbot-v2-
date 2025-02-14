import os
import faiss
import json
import torch
import numpy as np
import pandas as pd
import subprocess
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Limit CPU threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Check device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
data = pd.read_csv("/Users/shubhamfufal/Chatbot(v2)/MentalHealthChatbot-v2-/ESConv.csv")

# Clean text
def clean_text(text):
    return str(text).strip().replace("\n", " ")

for col in ["situation", "content"]:
    data[col] = data[col].apply(clean_text)

# Load JSON data
with open("/Users/shubhamfufal/Chatbot(v2)/MentalHealthChatbot-v2-/FailedESConv.json") as f:
    Data = json.load(f)

# Combine conversations
def combine_conversation(conv):
    context = f"Experience: {conv['experience_type']} | Emotion: {conv['emotion_type']}\n"
    context += f"Situation: {conv['situation']}\nDialogue:\n"
    for turn in conv['dialog']:
        role = turn['speaker'].upper()
        context += f"[{role}]: {turn['content'].strip()}\n"
    return context

conversations = [combine_conversation(conv) for conv in Data]

# Load embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
conversation_embeddings = np.load("/Users/shubhamfufal/Chatbot(v2)/MentalHealthChatbot-v2-/conversation_embeddings.npy")

# FAISS index
index = faiss.IndexFlatL2(conversation_embeddings.shape[1])
index.add(conversation_embeddings)

# Initialize emotion classifier pipeline (using the existing Hugging Face model)
emotional_model = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device
)

# Function to get emotion
def get_emotion(text):
    return emotional_model(text)[0]["label"]

# Retrieve context from the FAISS index
def retrieve_context(query, emotion, top_k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [conversations[idx] for idx in indices[0]]

# New function to call the Ollama model via subprocess
def generate_supportive_response(prompt):
    # The command will run the model "deepseek-r1:7b" via Ollama.
    # This assumes that your CLI is set up and that the model accepts the prompt via stdin.
    command = ["ollama", "run", "deepseek-r1:7b"]
    result = subprocess.run(command, input=prompt, capture_output=True, text=True)
    # You might need to adjust the parsing of result.stdout based on the CLI's output format.
    return result.stdout.strip()

# Main function to process user queries
def process_query(query):
    # Handle greetings
    if any(greeting in query.lower() for greeting in ["hi", "hello", "hey"]):
        return "Hello! I'm here to listen. How are you feeling today?"
    
    emotion = get_emotion(query)
    context = retrieve_context(query, emotion)[0]
    
    # Shorten context: take only the last 3 lines to avoid overwhelming the model
    context_lines = context.strip().splitlines()
    shortened_context = "\n".join(context_lines[-3:]) if len(context_lines) >= 3 else context
    
    # Build a refined prompt
    prompt = (
        f"Context:\n{shortened_context}\n\n"
        f"User Query: {query}\n\n"
        "Based on the above, please provide a supportive and empathetic response. "
        "Do not repeat the context or previous dialogue."
    )
    
    response = generate_supportive_response(prompt)
    return response.strip()

# Testh
print("Sample response:", process_query("I'm feeling a bit sad"))
