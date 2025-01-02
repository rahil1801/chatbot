import streamlit as st
import nltk
import numpy as np
from tensorflow.keras.models import load_model  # Use TensorFlow's Keras
from nltk.stem import WordNetLemmatizer
import pickle
import json
import random

# Load the trained model and related data
try:
    model = load_model('./model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop the app if the model fails to load

words = pickle.load(open('./texts.pkl', 'rb'))
classes = pickle.load(open('./labels.pkl', 'rb'))
with open('./intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

def preprocess_input(sentence):
    # Tokenize and lemmatize the input sentence
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    # Create a bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_intent(sentence):
    bag = preprocess_input(sentence)
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25  # Confidence threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by confidence
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def chatbot_response(user_input):
    intents_list = predict_intent(user_input)
    if intents_list:
        response = get_response(intents_list, intents)
        return response
    else:
        return "I didn't understand that. Could you please rephrase?"

# Streamlit UI Configuration
st.set_page_config(
    page_title="Chatbot Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .stTextInput input {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stSidebar {
        background-color: #2c3e50;
        color: white;
    }
    .stMarkdown {
        color: black;
    }
    .chat-history {
        background-color: #34495e;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-history:hover {
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🤖 Chatbot Assistant")

# Sidebar for Conversation History
with st.sidebar:
    st.title("Conversation History")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display chat history in a hoverable dialog box
    for chat in st.session_state['chat_history']:
        with st.container():
            st.markdown(
                f"""
                <div class="chat-history">
                    <strong>You:</strong> {chat['user']}<br>
                    <strong>Bot:</strong> {chat['bot']}
                </div>
                """,
                unsafe_allow_html=True
            )

    # Clear chat button
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state['chat_history'] = []

# Main Chat Interface
# Use a unique key for the text input to reset it
user_input_key = "user_input_" + str(len(st.session_state.get('chat_history', [])))
user_input = st.text_input("Type your message here...", "", key=user_input_key)

if st.button("Send", key="send_button"):
    if user_input.strip():
        # Get response from the chatbot
        response = chatbot_response(user_input)

        # Update chat history
        st.session_state['chat_history'].append({"user": user_input, "bot": response})

        # Rerun to update the UI and reset the input box
        st.rerun()

# Display the latest chat in the main UI
if st.session_state['chat_history']:
    latest_chat = st.session_state['chat_history'][-1]
    st.markdown(f"**You:** {latest_chat['user']}")
    st.markdown(f"**Bot:** {latest_chat['bot']}")