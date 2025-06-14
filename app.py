import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Model Selection
model_option = st.sidebar.selectbox(
    "Choose Model Variant:",
    ("LSTM + EarlyStopping", "GRU")
)

model_file_map = {
    "LSTM + EarlyStopping": "next_word_lstm_model_with_early_stopping.h5",
    "GRU": "next_word_lstm.h5"
}
model = load_model(model_file_map[model_option])


with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Next Word Prediction (LSTM + EarlyStopping)</h1>", unsafe_allow_html=True)
st.markdown("### Enter a sequence of words and let the LSTM model guess what comes next!")

input_text = st.text_input("Type your sentence below:", "Once upon a time, there ")

st.sidebar.markdown("### 🧠 Model Details")
if model_option == "LSTM + EarlyStopping":
    st.sidebar.write("- LSTM architecture with 2 layers")
    st.sidebar.write("- Includes EarlyStopping callback")
    st.sidebar.write("- More stable training")
elif model_option == "GRU":
    st.sidebar.write("- GRU (Gated Recurrent Unit) based model")
    st.sidebar.write("- Faster training, fewer parameters than LSTM")
    st.sidebar.write("- May generalize better on smaller datasets")


if st.button("Predict Next Word"):
    if input_text.strip():
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.success(f"**Next word prediction:** `{next_word}`")
    else:
        st.warning("Please enter a valid input.")


