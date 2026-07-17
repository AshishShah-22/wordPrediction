import os

# Suppress TensorFlow logs (must be before importing tensorflow)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pickle
import streamlit as st
import keras

keras.config.enable_unsafe_deserialization()
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_my_model():
    return load_model("next_Word_predition.h5")

Gru_model = load_model("Word_prediction_Gru.h5")


# -----------------------------
# Load Tokenizer (Cached)
# -----------------------------
@st.cache_resource
def load_my_tokenizer():
    with open("tokenizer.pkl", "rb") as handle:
        return pickle.load(handle)


model = load_my_model()
tokenizer = load_my_tokenizer()

# Create reverse dictionary once
index_word = {v: k for k, v in tokenizer.word_index.items()}


# -----------------------------
# Prediction Function for LSTM Model
# -----------------------------
def predict_next_word(text):
    max_sequence_len = model.input_shape[1] + 1

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return "No known words found."

    # Keep only last max_sequence_len-1 words
    token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    prediction = model.predict(token_list, verbose=0)

    predicted_index = np.argmax(prediction, axis=-1)[0]

    return index_word.get(predicted_index, "Word not found")


# -----------------------------
# Prediction Function for GRU Model
#------------------------------

def predict_next_word_gru(text):
    max_sequence_len = Gru_model.input_shape[1] +1
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list)==0:
        return "No known words found."
    token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences(
        [token_list],
        maxlen = max_sequence_len -1,
        padding = 'pre'
    )
    prediction = Gru_model.predict(token_list,verbose=0)
    predicted_index = np.argmax(prediction, axis=-1)[0]

    return index_word.get(predicted_index, )

# Streamlit UI

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠"
)

st.title("Next Word Prediction using LSTM")

input_text1 = st.text_input(
    "Enter a sentence",
    "To be or not to be"
)

if st.button("Predict Next Word1"):
    next_word1 = predict_next_word(input_text1)

    st.success(f"Predicted Next Word: **{next_word1}**")


st.title(" Next Word Prediction using GRU")

input_text2 = st.text_input(
    "Enter a sentence",

)

if st.button("Predict Next Word2"):
    next_word2 = predict_next_word_gru(input_text2)

    st.success(f"Predicted Next Word: **{next_word2}**")
