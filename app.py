import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = tf.keras.models.load_model('sentiment_model.h5')
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Use the same max_length from training
max_length = 28  # update as needed

st.title("Sentiment Analysis App")
user_text = st.text_input("Enter text for sentiment analysis:")

if st.button("Analyze"):
    seq = tokenizer.texts_to_sequences([user_text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    st.write("Predicted Sentiment:", sentiment)
