import streamlit as st
import nltk
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pickle
import json
import random

nltk.download('punkt_tab')
nltk.download('wordnet')

# Load the trained model and related data
model = load_model('./model.h5')
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
st.set_page_config(page_title="Chatbot", layout="centered")

# Sidebar Menu
st.sidebar.title("Menu")
menu_option = st.sidebar.radio(
    "Choose an option:",
    ("Chatbot", "Conversation History", "About the Chatbot")
)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Chatbot UI
if menu_option == "Chatbot":
    st.title("Talk Data to Me")
    st.markdown("Type a message below to interact with the chatbot.")

    user_input = st.text_input("Type your message", "", key="user_input")
    if st.button("Send"):
        if user_input:
            # Get response from the chatbot
            response = chatbot_response(user_input)

            # Update chat history
            st.session_state['chat_history'].append({"user": user_input, "bot": response})

    # Display chat history
    for chat in st.session_state['chat_history']:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

# Conversation History
elif menu_option == "Conversation History":
    st.title("Conversation History")
    if st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Bot:** {chat['bot']}")
        
        # Clear chat history button
        if st.button("Clear History"):
            st.session_state['chat_history'] = []
            st.success("Chat history cleared!")
    else:
        st.info("No conversation history available.")

# About the Chatbot
elif menu_option == "About the Chatbot":
    st.title("About the Chatbot")
    st.markdown("""
    ### Intent-Based Chatbot
    This chatbot is an **intent-based chatbot** designed to understand and respond to user queries based on predefined intents. 
    It uses natural language processing (NLP) techniques to classify user input into specific intents and generate appropriate responses.

    #### How It Works:
    1. **Intent Recognition**: The chatbot uses a trained machine learning model to classify user input into one of the predefined intents.
    2. **Response Generation**: Based on the recognized intent, the chatbot selects a random response from the available options for that intent.
    3. **Conversation History**: All interactions are stored in a session-based conversation history, which can be viewed or cleared.

    #### Technologies Used:
    - **Natural Language Toolkit (NLTK)**: For tokenization and lemmatization.
    - **Keras**: For building and training the intent classification model.
    - **Streamlit**: For creating the interactive web interface.

    #### Developer:
    This chatbot was developed as part of a project to demonstrate the capabilities of intent-based conversational agents.
    """)