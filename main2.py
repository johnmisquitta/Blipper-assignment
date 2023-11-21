!pip install speech_recognition

import streamlit as st
import speech_recognition as sr

# Initialize the speech recognizer
r = sr.Recognizer()

def recognize_speech():
    try:
        # Record audio from the microphone
        with sr.Microphone() as source:
            audio = r.listen(source)

        # Recognize the speech
        transcription = r.recognize_google(audio)

        # Update the transcription display
        st.write("Transcription:", transcription)
    except sr.UnknownValueError:
        st.write("Could not understand audio")
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service; {e}")

# Create the Streamlit app
st.title("Speech Recognition Demo")

# Add a button to start recognition
if st.button("Start Recognition"):
    recognize_speech()
