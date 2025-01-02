# chatbot
# Please read this file first!!

In this project, we are creating an intent based chatbot which recognizes the intents of the user and gives answer accordingly. This chatbot is built using NLP (Natural Language Processing) techniques/library and Keras (Deep learning Framework) to extract the intents from the dataset. Keras is an open source library that provides a Python interface for ANN (Artifical Neural Network). The chatbot is built using Streamlit, a python library for building interactive web interfaces.

# To get started with the project, follow these steps:
Install the required Python libraries using:
pip install -r requirements.txt

Run the application using:
streamlit run app.py

This will host the chatbot on locahost and will be alloted a specific PORT NUMBER.

# DATASET!!!
The dataset named as (intents.json) consists of following things:

Intents: The intent of the user input (e.g. "greeting", "budget", "about")
Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
Text: The user input text.
Note: If the required packages are outdated or did not accept the newer version, you can manually install the specific packages using:

pip install <package_name>