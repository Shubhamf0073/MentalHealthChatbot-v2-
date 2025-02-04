# app.py

import os
import json
import numpy as np
import streamlit as st
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer

# ------------------------------
# Helper: Clean and Combine Conversations
# ------------------------------

def clean_text(text):
    text = str(text).strip()
    text = text.replace("\n", " ")
    return text

def combine_conversation(conv):
    context = (
        f"Experience: {conv['experience_type']} | Emotion: {conv['emotion_type']} | Problem: {conv['problem_type']} \n"
        f"Situation: {conv['situation']}\nDialogue:\n"
    )
    for turn in conv['dialog']:
        role = turn['speaker'].upper()
        extra = ""
        if role == "LISTENER":
            if 'strategy' in turn.get('annotation', {}):
                extra += f" [Strategy: {turn['annotation']['strategy']}]"
            if 'feedback' in turn.get('annotation', {}):
                extra += f" [Feedback: {turn['annotation']['feedback']}]"
        context += f"[{role}]{extra}: {turn['content'].strip()}\n"
    return context

# ------------------------------
# Load Data and Build FAISS Index
# ------------------------------

@st.cache_resource
def load_conversations_and_index():
    # Load the FailedESConv JSON file (assumes file is in the repo root)
    json_path = "FailedESConv.json"
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Create a list of conversation strings using combine_conversation()
    conversations = [combine_conversation(conv) for conv in raw_data]

    # Load precomputed conversation embeddings.
    # Assumes the embeddings file is named "conversation_embeddings.npy" and is in the repo root.
    embeddings_path = "conversation_embeddings.npy"
    conversation_embeddings = np.load(embeddings_path)

    # Create FAISS index using L2 (Euclidean) distance.
    dimension = conversation_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(conversation_embeddings))

    return conversations, index

conversations, faiss_index = load_conversations_and_index()

# ------------------------------
# Load Models and Pipelines
# ------------------------------

@st.cache_resource
def load_models():
    # Load the sentence transformer for encoding
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Pipelines for emotion detection, summarization, and response generation.
    emotional_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    generator = pipeline(
        "text-generation",
        model="EleutherAI/gpt-neo-2.7B",
        torch_dtype="auto"  # Use auto or torch.float16 if on GPU
    )

    # Tokenizer for prompt trimming
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    return embedder, emotional_model, summarizer, generator, tokenizer

embedder, emotional_model, summarizer, generator, tokenizer = load_models()

# ------------------------------
# Define Core Functions
# ------------------------------

def get_emotion(text):
    """Detects the emotion of the input text."""
    result = emotional_model(text)
    return result[0]["label"]

def retrieve_context(query, top_k=1):
    """
    Retrieve matching context from the FAISS index.
    We encode the query using the same embedder and then search the index.
    """
    query_embedding = embedder.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    retrieved = [conversations[idx] for idx in indices[0]]
    return retrieved

def summarize_context(context, max_length=50):
    """
    Summarize the retrieved context.
    If context is a list, join into one text.
    """
    if isinstance(context, list):
        context_text = "\n".join(context)
    else:
        context_text = context
    summary = summarizer(context_text, max_length=max_length, min_length=25, do_sample=False)
    return summary[0]['summary_text']

def trim_prompt(prompt, max_tokens=512):
    """
    Trim the prompt to a maximum number of tokens.
    """
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    return tokenizer.decode(tokens, skip_special_tokens=True)

def generate_response(context, query, emotion):
    """
    Generate a response by summarizing the context and building a prompt.
    """
    # Summarize context for brevity.
    context_summary = summarize_context(context, max_length=20)
    prompt = (
        f"Context: {context_summary}\n"
        f"User Query: {query}\n"
        f"Emotion: {emotion}\n\n"
        "Generate a concise, empathetic response (2-3 sentences) that validates the user's feelings and offers supportive advice.\n"
        "Answer:"
    )
    prompt = trim_prompt(prompt, max_tokens=512)
    response = generator(
        prompt,
        max_new_tokens=100,
        num_return_sequences=1,
        truncation=True,
        temperature=0.95,
        top_p=0.9
    )
    generated_text = response[0]["generated_text"]
    if "Answer:" in generated_text:
        generated_text = generated_text.split("Answer:", 1)[-1].strip()
    return generated_text

def collect_feedback(response_text, user_rating):
    """
    Save the response along with user feedback for future analysis.
    """
    feedback_data = {"response": response_text, "user_rating": user_rating}
    with open("feedback.json", "a") as f:
        json.dump(feedback_data, f)
        f.write("\n")

def process_query(query, user_rating=None):
    """
    Process the user's query: detect emotion, retrieve context, generate response,
    and optionally collect feedback.
    """
    # Step 1: Emotion detection.
    emotion = get_emotion(query)
    st.write(f"[ERA] Detected Emotion: {emotion}")

    # Step 2: Retrieve context using FAISS.
    context = retrieve_context(query, top_k=1)
    st.write(f"[CRA] Retrieved Context:\n{context}")

    # Step 3: Summarize context.
    context_summary = summarize_context(context, max_length=50)
    st.write(f"[Summary] Context Summary: {context_summary}")

    # Step 4: Generate response.
    response_text = generate_response(context, query, emotion)
    st.write(f"[RGA] Generated Response:\n{response_text}")

    # Step 5: Collect feedback if provided.
    if user_rating is not None:
        collect_feedback(response_text, user_rating)
        st.write("[LA] Feedback collected.")

    return response_text

# ------------------------------
# Streamlit UI Components
# ------------------------------

st.title("Emotional Support Chatbot")
st.write("Hi Shubham! How are you feeling today?")

query = st.text_input("Enter your query here:")
user_rating = st.number_input("Rate the response (optional, 1-5):", min_value=1, max_value=5, step=1, format="%d")

if st.button("Get Response") and query:
    final_response = process_query(query, user_rating if user_rating else None)
    st.subheader("Response:")
    st.write(final_response)
