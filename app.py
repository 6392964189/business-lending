import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import os

class ColdCallBot:
    def __init__(self, model_path, tokenizer):
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = tokenizer
    
    def generate_response(self, text):
        sequences = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=7, padding='post')  # Adjusted padding length
        print("Padded Sequence Shape:", padded.shape)  # Debug line
        prediction = self.model.predict(padded)
        return prediction

# Sample data
sentences = [
    'Hello, I am interested in business lending.',
    'Can we schedule an appointment?',
    'I want to join the webinar.',
    'What are your business lending rates?',
    'Tell me more about your services.'
]
labels = [1, 1, 1, 0, 0]  # 1 for positive responses, 0 for neutral/informative

# Tokenization
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')

# Convert lists to numpy arrays
padded_sequences = np.array(padded_sequences)
labels = np.array(labels)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=padded_sequences.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=30)

# Save model
model_file_path = 'cold_call_bot.h5'
if os.path.exists(model_file_path):
    os.remove(model_file_path)
model.save(model_file_path)

# Load model and tokenizer
bot = ColdCallBot(model_file_path, tokenizer)

# Streamlit UI
st.title("Cold Calling Bot")
user_input = st.text_input("Ask something related to business lending:")
if st.button("Submit"):
    response = bot.generate_response(user_input)
    st.write(response)
