
import streamlit as st


st.header("Pick an Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"]) #upload file
st.audio(uploaded_file, format='audio/wav') #display audio
st.write("<hr>", unsafe_allow_html=True) 
