import streamlit as st
import sounddevice as sd
import wave
from pyannote.audio import Pipeline
import io
import tempfile  # Import the tempfile module to handle temporary files
import soundfile as sf
import torch
import torchaudio
import librosa
import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline
from pydub import AudioSegment
import difflib

from transformers import AutoTokenizer, T5ForConditionalGeneration
import os
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import difflib

from transformers import AutoTokenizer, T5ForConditionalGeneration ,AutoModelForSequenceClassification


import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline
import os
from transformers import pipeline


# To shift the processing from cpu to gpu
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")
    
#hugging face pipelines
    
#pipline for summarization
summarizer = pipeline("summarization", model="knkarthick/meeting-summary-samsum")
# pipline for threat detection
threat_classifier = pipeline("text-classification", model="grammarly/detexd-roberta-base")
# pipline for sales pitch detection
sales_classifier = pipeline("text-classification", model="hannoh/03_model_sales")
sales_tokenizer = AutoTokenizer.from_pretrained("obsei-ai/sell-buy-intent-classifier-bert-mini")
sales_model = AutoModelForSequenceClassification.from_pretrained("obsei-ai/sell-buy-intent-classifier-bert-mini")
#pipeline for emotion detection
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

#formating of the page to full screen
st.set_page_config(layout="wide")

#to save tempoary audio file so it can be deleted after processing
temp_file=None

#to store dictionary for futher processinng like analytics
dictionary={}
''' Example
    "speaker": "SPEAKER_00",
    "speaker_name": "Speaker 1",
    "start_time": 20.16129,
    "end_time": 21.655348,
    "gap": 0.3225800000000021,
    "file_name": "audio_segments\\SPEAKER_00_20.16-21.66.wav",
    "transcription": " 4977.",
    "emotion": "neutral 😐",
    "intent": "switch_account ",
    "colour": "#FFE15D"
''' 

# to store summary so it can be futher divided into speaker1 speakr2
summary=[]

from datetime import datetime

# to detect sensitive content
def threat_detection(text: str):
    scores = threat_classifier(text, top_k=None)
    return scores[0]
# to detdct sellers pitch
def sales_detection(text: str):
    inputs = sales_tokenizer(text, return_tensors="pt")
    outputs = sales_model(**inputs)
    logits = outputs.logits
    softmax_probs = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(softmax_probs, dim=1).item()
    class_labels = ["sell", "nutral"]
    predicted_label = class_labels[predicted_class]
    print(f"Predicted Class: {predicted_label}")
    print(f"Class Probabilities: {softmax_probs.tolist()}")
    if softmax_probs[0][0] > softmax_probs[0][1]:
        return("SELL")
    else:
        return("NUTRAL")

def time_to_seconds(time_str):
    time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
    seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    return seconds
# to detect what customer wants
def intent_detection(text):
    from transformers import pipeline
    classifier = pipeline("text-classification", model="vineetsharma/customer-support-intent-albert", return_all_scores=True)
    intent = None
    ner_results = classifier(text)  # Analyze the entity for the current conversation starter
    if ner_results and ner_results[0]:
        label_with_highest_score = max(ner_results[0], key=lambda x: x['score'])
        intent = f"{label_with_highest_score['label']} "
    else:
        intent = "No intent detected."

    return intent
# detect classification of enteties such as phone no email ids names organization names
def entity_detection(text):
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline
    tokenizer = AutoTokenizer.from_pretrained("mdarhri00/named-entity-recognition")
    model = AutoModelForTokenClassification.from_pretrained("mdarhri00/named-entity-recognition")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)    
    ner_results = nlp(text)
    # Initialize a dictionary to store unique words for each entity type
    unique_words = {}
    for entity in ner_results:
        entity_type = entity['entity']
        word = entity['word']
        # Check if the word is a subword token and needs to be merged
        if word.startswith('##'):
            word = word[2:]  # Remove '##' prefix
        # Add the word to the set of unique words 
        unique_words.setdefault(entity_type, set()).add(word)
    entity_dict = {}
    for entity_type, words in unique_words.items():
        word_list = list(words)[:5] 
        entity_dict[entity_type] = word_list
    return entity_dict

# detect emotion of a sentence
def emotion_detection(text):
    from transformers import pipeline
    result = classifier(text)
    # Extract the emotion scores from the 'scores' key
    emotion_scores = result[0]

    # Find the lable with highest score
    highest_emotion = max(emotion_scores, key=lambda x: x['score'])
    emoji_mapping = {
        'anger': '🤬',
        'disgust': '🤢',
        'fear': '😨',
        'joy': '😀',
        'neutral': '😐',
        'sadness': '😭',
        'surprise': '😲'
    }

    # Print the highest score and corresponding emoji
    result=(f"{highest_emotion['label']} {emoji_mapping[highest_emotion['label']]}")
    print(result)
    return result

# to highlight changed words after grammar correction
def highlight_changed_words(text1, text2):
    differ = difflib.Differ()
    diff = list(differ.compare(text1.split(), text2.split()))
    highlighted_text = []
    for item in diff:
        if item.startswith(' '):
            highlighted_text.append(item[2:])
         elif item.startswith('+'):
            highlighted_text.append(f'<span style="background-color: #aaffaa;">{item[2:]}</span>')

    return ' '.join(highlighted_text)

#corrects the misspelled words
def grammar_correct(text):
    tokenizer = AutoTokenizer.from_pretrained("pszemraj/grammar-synthesis-small")
    model = T5ForConditionalGeneration.from_pretrained("pszemraj/grammar-synthesis-small")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=1000)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# give different colour to different speaker used during printig on web page
def convert_speaker_name(speaker_name):
        my_list = ["#FFE15D","#FD841F", "blue", "green", "Purple"]
        if speaker_name.startswith("SPEAKER_"):
            try:
                index = int(speaker_name.split("_")[1])
                changed_speaker_name=f"Speaker {index + 1}"
                colour=my_list[index]
                return changed_speaker_name,colour
            except ValueError:
                return speaker_name,"yellow"
        else:
            return speaker_name,"yellow"

# read diarization results from rttm file
def read_rttm_file(filename):
    """Reads an RTTM file and returns a list of dictionaries, where each dictionary contains the following information:

    speaker
    start_time
    end_time
    
    Example
    0.619694 2.249576 SPEAKER_00 1.0 1.0 <NA> <NA>
    3.268251 7.207131 SPEAKER_00 1.0 1.0 <NA> <NA>
    8.395586 17.954160 SPEAKER_00 1.0 1.0 <NA> <NA>
    19.057725 19.838710 SPEAKER_00 1.0 1.0 <NA> <NA>
    """
    
    #reads rttm file and fetches start time end time and speaker id
    with open(filename, "r") as f:
        lines = f.readlines()
    utterances = []
  
    previous_end_time = 0.0  # Initialize the previous_end_time
    output_dir = "audio_segments" # to create a directory
    os.makedirs(output_dir, exist_ok=True)    

    # pipeline for transcription from hugging face
    pipeline = FlaxWhisperPipline("openai/whisper-tiny")#, dtype=jnp.bfloat16, batch_size=16
    # iterates through diarization results
    for line in lines:
        speaker = line.split()[2] #3rd position of rttm file # end time
        start_time = float(line.split()[0]) #1st position of rttm file #start time
        end_time = float(line.split()[1]) #2nd poition of rttm file
        gap = start_time - previous_end_time  # Calculate the latency
        
        # to create small chunks of audio clips and store in directory used futher in transcription
        segment_path = os.path.join("audio_segments", f"{speaker}_{start_time:.2f}-{end_time:.2f}.wav") #file name
        #st.write(segment_path)
        audio = AudioSegment.from_file(temp_file,format="wav")  
        print(audio)

        start_time_ms = start_time*1000  # convert minutes into mili seconds
        end_time_ms = end_time*1000
        
        # segment the audio into smaller clips and save it into directory
        cut_audio = audio[start_time_ms:end_time_ms]
        cut_audio.export(segment_path, format="wav")
        
        # transcribe audio and save it in transcription dictionary
        transcription=None
        try:
            transcription = pipeline(segment_path, task="transcribe")
        except ValueError as e:
            transcription = {"text": ""}
        print(transcription)
        # assigning colour to each speaker
        speaker_name,colour=convert_speaker_name(speaker)
        
        # append transcription in the global dictionary
        global transcript
        transcript.append(speaker_name)
        transcript.append(transcription['text']) #transcribe 
        
        # this is to create a dictionary according to timestamp  
        utterances.append({
            "speaker": speaker,
            "speaker_name":speaker_name,
            "start_time": start_time,
            "end_time": end_time,
            "gap": gap,  # add latency
            "file_name":segment_path, # append clipped file name so it could be read futher
            "transcription":transcription['text'], #append transcription to dictionary
            "emotion": emotion_detection(transcription['text']), #append emotion to dictionary
            "intent":intent_detection(transcription['text']),#append intent to dictionary
            "colour":colour #append colour to dictionary called from function

        })
        


        
        # making previous time as end time
        previous_end_time = end_time
    global dictionary
    # saving all values like emotion detection intent detection transcription to global dictionary
    dictionary= utterances
    return utterances



#Code starts from here
############################################################################################################
st.header("Pick an Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"]) #upload file with file picker
st.audio(uploaded_file, format='audio/wav') #display audio
st.write("<hr>", unsafe_allow_html=True) 


if uploaded_file: #check audio is valid
    audio_data, sample_rate = sf.read(io.BytesIO(uploaded_file.read()))

    # Create a temporary audio file to save the audio_data after processing it will get deleted
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        sf.write(temp_audio_path, audio_data, sample_rate)
    
    # 
   
    from pyannote.audio import Pipeline


     #pipeline for diarization from hugging face
    pipeline = Pipeline.from_pretrained(
         # backup models each one has different processing time
        #pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2022.07")#,use_auth_token="API_KEY")
         #pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",use_auth_token="API_KEY")
         #pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",use_auth_token="API_KEY")
        "pyannote/speaker-diarization-3.0",use_auth_token="API_KEY")


    #pipeline.to(torch.device("cuda"))
    # send pipeline to GPU (when available)
    # apply pretrained pipeline
    waveform, sample_rate = torchaudio.load(temp_audio_path)
    #convert audio into bits
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    print(diarization)
    st.write(diarization)

    # write diarization result into rttm file
    with open('audio.rttm', "w") as rttm_file:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = turn.start  # Use start time in seconds
            end_time = turn.start + turn.duration
            #iterate through diarization result and write in rttm file in this format
            rttm_file.write(f"{start_time:.6f} {end_time:.6f} {speaker} 1.0 1.0 <NA> <NA>\n")

        
    rttm_file_path = "audio.rttm"
    temp_file=temp_audio_path
    utterances = read_rttm_file(rttm_file_path)
    


#st.write(dictionary)
# create layout  to divide screen in two parts
col1, col2 = st.columns([1, 1])
col1.header("Transcript")

for i in dictionary:
    speaker_number = i["speaker_name"].split()
    # this will print speaker number with assigned colour eg speker1 yellow speaker 2 red
    col1.markdown(
        #css for formatting
        #prints speakers Names

        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;width: 30px; height: 30px; background-color: {i['colour']}; border-radius: 50%; margin-right: 12px;margin-top: 25px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;"> {speaker_number[1]}</div>
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;"> {i['speaker_name']}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    #prints timestamp like start time end time and latency emotion detection result
    col1.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem; font-weight: 400; line-height: 150%; font-family: AvertaStd, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol; margin-left: 32px; color: #007affe6;margin-left: 42px;">
                {int(i['start_time'])} sec - {int(i['end_time'])} sec | Latency ~ {i['gap']:.1f} sec
            </div>
        </div>


        <div style="font-size: 1.2rem; font-weight: italic; line-height: 150%; font-family: AvertaStd, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol; margin-left: 32px; color: grey;margin-left: 42px;">
            Intent ~ {i['intent']} | Emotion ~ {i['emotion']}
        </div>
        ''',
        unsafe_allow_html=True
    )
    #prints speakers timestamp
    col1.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-kerning: none; font-weight: 400; line-height: 150%; font-family:AvertaStd, MonumentGrotesk, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol; margin-left: 42px; color: black;">
                {i['transcription']}
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

col2.header("Grammar Correction")

for i in dictionary:
    speaker_number = i["speaker_name"].split()
    #prints speakers Names

    col2.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;width: 30px; height: 30px; background-color: {i['colour']}; border-radius: 50%; margin-right: 12px;margin-top: 25px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;"> {speaker_number[1]}</div>
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;"> {i['speaker_name']}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    #prints speakers timestamp

    col2.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem; font-weight: 400; line-height: 150%; font-family: AvertaStd, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol; margin-left: 32px; color: #007affe6;margin-left: 42px;">
                {int(i['start_time'])} sec - {int(i['end_time'])} sec | Latency ~ {i['gap']:.1f} sec
            </div>
        </div>                

        <div style="font-size: 1.2rem; font-weight: italic; line-height: 150%; font-family: AvertaStd, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol; margin-left: 32px; color: grey;margin-left: 42px;">
            Intent ~ {i['intent']} | Emotion ~ {i['emotion']}
        </div>
        ''',
        unsafe_allow_html=True
    )
    #prints corrected text using grammer correction
    diff_text = highlight_changed_words(i['transcription'], grammar_correct(i['transcription']))
    col2.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-kerning: none; font-weight: 400; line-height: 150%; font-family:AvertaStd, MonumentGrotesk, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol; margin-left: 42px; color: black;">
             {diff_text}
            </div>
        </div>
        ''',
        unsafe_allow_html=True

    )

st.write("<hr>", unsafe_allow_html=True)

# using pandas to manupalate data and visvulaize data
import pandas as pd
st.header("Insight")
#create dataframe from dictionary
df = pd.DataFrame(dictionary)
#find occurance of words
def count_occurrences(text, word):
        return text.str.count(word).sum()


#print summary of users called from summary model
def summarize_text(text):
    result = summarizer(text)
    return result[0]['summary_text']


col1, col2 = st.columns(2)
with col1:
    #Text Label
    col1.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">SUMMARY</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    #grouping both speakers transcript
    combined_transcripts = df.groupby('speaker_name')['transcription'].apply(' '.join).reset_index()
    speaker_summary = []
    # to seprate transcription of individual speakers 
    for index, row in combined_transcripts.iterrows():
        speaker_name = row['speaker_name']
        transcript = row['transcription']
        speaker_summary.append({
            "speaker": speaker_name,
            "summary": transcript,
        })
    # to fetch summary of individual speakers 
    for i in speaker_summary:
        col1.info(f"{i['speaker']} : {summarize_text(i['summary'])}")

    #Count total words by both speaker
    df['word_count'] = df['transcription'].str.split().str.len()
    total_word_count = df.groupby('speaker_name')['word_count'].sum().reset_index()
    
    
    # create a pie chart of words spoken by each speaker
    fig = px.pie(df, values='word_count', names='speaker_name', title='Word Count Distribution by Speaker')
    fig.update_traces(textinfo='percent+label', marker=dict(colors=px.colors.qualitative.Set3), textposition='inside')
    fig.update_layout(legend_title="Regions", legend_y=0.9)
    fig.update_layout(width=350, height=350)
    col1.plotly_chart(fig)#, use_container_width=True)

with col2:
    #Text Label
    col2.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">
            TOTAL WORDS</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    # print total words on webpage
    df['word_count'] = df['transcription'].str.split().str.len()
    total_word_count = df['word_count'].sum()
    col2.info(f"{total_word_count} words.")
    
    #Text Label
    col2.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">
            TOTAL DURATION</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    # print total duartion of minutes both speaker has spoken
    total_duration_minutes = (df['end_time'] - df['start_time']).sum() / 60
    col2.info(f"Total Duration in Minutes: {total_duration_minutes:.2f} minutes")

    #Text Label
    col2.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">
            FIND WORD</div>
        </div>
        ''',
        unsafe_allow_html=True
    )


    #Takes user input from user to find word
    word_to_find = col2.text_input("")
    #Find occurance of words and prints
    if word_to_find:
        occurrences = df.groupby('speaker_name')['transcription'].apply(count_occurrences, word=word_to_find)

        for speaker_name, count in occurrences.items():
            col2.info(f"{speaker_name} has spoken The word '{word_to_find}' {count} times.")
    # print basic information like wpm ,total duration ,total latency ,total words
    col3, col4, col5, col6, col7 = st.columns(5)
    df['word_count'] = df['transcription'].str.split().str.len()
    word_count = df['transcription'].str.split().str.len()
    #Text Lable
    col3.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">
            SPEAKER</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    #Text Lable
    col4.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">
            WORDS</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    #Text Lable
    col5.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">
            SPEED</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    #Text Lable
    col6.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">
            DURATION</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    #Text Lable
    col7.markdown(
    '''
    <div style="display: flex; align-items: center;">
        <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">
            Latency
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)

# Print the words spoken by each speaker
speaker_word_counts = df.groupby('speaker_name')['word_count'].sum().reset_index()
for index, row in speaker_word_counts.iterrows():
    # Print speaker id
    col3.info(f"{row['speaker_name']}")
    # Print speaker word count
    col4.info(f"{row['word_count']} words.")

for speaker_name, group in df.groupby('speaker_name'):
    total_duration = (group['end_time'] - group['start_time']).sum() / 60  # in minutes
    total_words = group['transcription'].str.split().apply(len).sum()
    wpm = total_words / total_duration
    # Print speaker wpm
    col5.info(f"{wpm:.2f} wpm")
    # Print speaker duration by individual speaker
    col6.info(f"{total_duration:.2f} minutes")


# Print total latency by individual speaker
total_gap_speaker = df.groupby('speaker_name')['gap'].sum().reset_index()
for index, row in total_gap_speaker.iterrows():
    col7.info(f"{row['gap']:.2f} sec.")


full_transcript = " ".join(df['transcription'])
import json
#Text Label
col1.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">
            Entity Detection</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
# result of entity detection
sample_dict_str = json.dumps(entity_detection(full_transcript))
for entity_type, words in entity_detection(full_transcript).items():
    with col1.expander(entity_type):
        for i, word in enumerate(words):
            st.write(f"{i}: \"{word}\"")

#Text Label
col1.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">
            Threat Detection</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
# result of Threat detection
with col1.expander("statements"):
    for label in dictionary:
        #col2.write(label['transcription'])
        text=threat_detection(label['transcription'])
        #col2.write(text['label'])
        if text['label'] == "LABEL_5":
            st.write(f"{label['transcription']} intensity: very high risk")
            
        if text['label'] == "LABEL_4":
            st.write(f"{label['transcription']} intensity: high risk")
            
        if text['label'] == "LABEL_3":
            st.write(f"{label['transcription']} intensity:medium risk")
# Text Label
col1.markdown(
        f'''
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem;font-weight: bold; font-family:AvertaStd, MonumentGrotesk, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;margin-right: 10px;margin-top: 20px;">
            Sales Pitch</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
# result of sales Pitch detection

with col1.expander("statements"):

    for label in dictionary:
            #col2.write(label['transcription'])
            text=sales_detection(label['transcription'])
            #st.write(text)
            if text == "SELL":
                st.write(f"{label['transcription']}")
