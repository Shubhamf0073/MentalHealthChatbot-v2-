import os
import faiss
import json
import torch
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Limit CPU threads (if running on CPU)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
data = pd.read_csv("/Users/shubhamfufal/Chatbot(v2)/MentalHealthChatbot-v2-/ESConv.csv")

# Clean text
def clean_text(text):
    text = str(text).strip()
    text = text.replace("\n", " ")
    return text

for col in ["situation", "content"]:
    data[col] = data[col].apply(clean_text)

# Load JSON data
json_path = "/Users/shubhamfufal/Chatbot(v2)/MentalHealthChatbot-v2-/FailedESConv.json"
with open(json_path) as f:
    Data = json.load(f)

# Combine conversations
def combine_conversation(conv):
    context = f"Experience: {conv['experience_type']} | Emotion: {conv['emotion_type']} | Problem: {conv['problem_type']} \n"
    context += f"Situation: {conv['situation']}\nDialogue:\n"
    for turn in conv['dialog']:
        role = turn['speaker'].upper()
        extra = ""
        if role == "LISTENER":
            if 'strategy' in turn['annotation']:
                extra += f" [Strategy: {turn['annotation']['strategy']}]"
            if 'feedback' in turn['annotation']:
                extra += f" [Feedback: {turn['annotation']['feedback']}]"
        context += f"[{role}]{extra}: {turn['content'].strip()}\n"
    return context

conversations = [combine_conversation(conv) for conv in Data]

# Load embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
conversation_embeddings = np.load("/Users/shubhamfufal/Chatbot(v2)/MentalHealthChatbot-v2-/conversation_embeddings.npy")

# FAISS index
dimension = conversation_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(conversation_embeddings)

# Pinecone setup
pc = Pinecone(api_key="pcsk_7UjC6j_G3F7vu1GjfD9MQpNNFmkSBieGQBmfsK29JqyvZK23aimzbzG1AKrSbf9aefGeaa")
index_name = "emotional-support"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="euclidean",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)
ids = [f"conv_{i}" for i in range(len(conversation_embeddings))]
vectors = list(zip(ids, conversation_embeddings.tolist()))
index.upsert(vectors)

# Define greetings
greetings = ["hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon", "good evening"]

# Initialize models
emotional_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
generator = pipeline(
    "text-generation",
    model="distilgpt2",  # Smaller model
    device=0 if device == "mps" else -1,  # Use MPS if available
)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Functions
def get_emotion(text):
    result = emotional_model(text)
    return result[0]["label"]

def retrieve_context(query, emotion, top_k=1):
    query_embedding = model.encode([query]).tolist()
    result = index.query(query_embedding, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in result["matches"]]

def summarize_context(context, max_length=50):
    summary = summarizer(context, max_length=max_length, min_length=25, do_sample=False)
    return summary[0]['summary_text']

def trim_prompt(prompt, max_tokens=512):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    return tokenizer.decode(tokens, skip_special_tokens=True)

def generate_response(context, query, emotion):
    context_summary = summarize_context(context, max_length=20)
    prompt = (
        f"{context_summary}\n"
        f"{query}\n"
        f"{emotion}\n\n"
        "Generate a concise, empathetic response (2-3 sentences) that validates the user's feelings and offers supportive advice.\n"
        "Answer:"
    )
    prompt = trim_prompt(prompt, max_tokens=512)
    response = generator(
        prompt,
        max_new_tokens=30,  # Reduce the number of tokens
        num_return_sequences=1,
        truncation=True,
        temperature=0.95,
        top_p=0.9
    )
    generated_text = response[0]["generated_text"]
    if "Answer:" in generated_text:
        generated_text = generated_text.split("Answer:", 1)[-1].strip()
    return generated_text

def collect_feedback(response, user_rating):
    feedback_data = {"response": response, "user_rating": user_rating}
    with open("feedback.json", "a") as f:
        json.dump(feedback_data, f)
        f.write("\n")

def process_query(query, user_rating=None):
    if any(greeting in query.lower() for greeting in greetings):
        return "Hello! How are you feeling today? If you're comfortable, please share what's on your mind."
    
    emotion = get_emotion(query)
    context = retrieve_context(query, emotion, top_k=1)
    context_summary = summarize_context(context, max_length=50)
    response = generate_response(context, query, emotion)
    
    if user_rating is not None:
        collect_feedback(response, user_rating)
    
    return response

# Test the chatbot
response = process_query("I'm feeling really low today.")
print(response)