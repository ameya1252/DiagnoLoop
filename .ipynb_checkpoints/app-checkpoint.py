from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import threading
import openai
import os
from flask_cors import CORS

# --------------------- NEW IMPORTS ---------------------
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import json

# --------------------- FLASK APP SETUP ---------------------
app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

# --------------------------------------------------------------------------
# 1) LOAD THE DATA & PREPROCESSING SETUP
# --------------------------------------------------------------------------
train_csv_path = "./symptom-disease-train-dataset.csv"
test_csv_path = "./symptom-disease-test-dataset.csv"
mapping_json_path = "./mapping.json"

# Load the CSV files; they have columns "text" and "label"
df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)

# ----------------------------
# 2) RE-MAP LABELS TO CONTIGUOUS INDICES
# ----------------------------
unique_labels = sorted(df_train["label"].unique())
label_to_index = {label: i for i, label in enumerate(unique_labels)}
index_to_label = {i: label for label, i in label_to_index.items()}

# Map the labels in the dataframes
df_train["new_label"] = df_train["label"].map(label_to_index)
df_test["new_label"] = df_test["label"].map(label_to_index)

# For training and evaluation (and later inference), use these:
X_train_text = df_train["text"]
y_train = df_train["new_label"]

X_test_text = df_test["text"]
y_test = df_test["new_label"]

# ----------------------------
# 3) LOAD THE MAPPING & SET UP REVERSE MAPPING
# ----------------------------
# mapping.json maps disease names to original label codes, e.g.:
# {"(Vertigo) Paroymsal  Positional Vertigo": 1047, "Abdominal Aortic Aneurysm": 207, ...}
with open(mapping_json_path, 'r') as f:
    disease_mapping_dict = json.load(f)

# Create a reverse mapping: from original label code to disease name.
reverse_mapping = {value: key for key, value in disease_mapping_dict.items()}

# ----------------------------
# 4) TEXT PREPROCESSING WITH TEXT VECTORIZATION
# ----------------------------
max_features = 20000  # Maximum number of tokens to consider
vectorizer = keras.layers.TextVectorization(
    max_tokens=max_features,
    output_mode='tf_idf'
)

# Adapt the vectorizer on the training text
vectorizer.adapt(X_train_text)

# Transform the training and test text into numeric vectors (if needed elsewhere)
X_train_vectorized = vectorizer(X_train_text)
X_test_vectorized = vectorizer(X_test_text)

# --------------------------------------------------------------------------
# 5) LOAD THE TRAINED KERAS MODEL
# --------------------------------------------------------------------------
# Make sure the saved model was trained on the vectorized text.
MODEL_PATH = "./trained_model1.h5"

try:
    keras_model = keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Successfully loaded Keras model from {MODEL_PATH}")
except Exception as e:
    print("[ERROR] Could not load Keras model:", e)
    keras_model = None

# --------------------------------------------------------------------------
# 6) OPENAI SETUP
# --------------------------------------------------------------------------
openai.api_key = os.environ.get("OPENAI_API_KEY", "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")

# --------------------------------------------------------------------------
# 7) GLOBALS FOR AUDIO TRANSCRIPTION
# --------------------------------------------------------------------------
recording = False
transcript = ""

# --------------------------------------------------------------------------
# 8) GPT-BASED KEY INFORMATION EXTRACTION
# --------------------------------------------------------------------------
def extract_key_information(conversation_text):
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that extracts key medical information from a doctor-patient conversation."
        },
        {
            "role": "user",
            "content": (
                f"Extract the patient's reported symptoms, duration, relevant medical history, "
                f"and any other important details from the following transcript:\n\n{conversation_text}"
            )
        }
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.3
        )
        key_info = response['choices'][0]['message']['content'].strip()
        return key_info
    except Exception as e:
        print("OpenAI API error:", e)
        return "Error extracting key information."

# --------------------------------------------------------------------------
# 9) AUDIO TRANSCRIPTION FUNCTION
# --------------------------------------------------------------------------
def transcribe_audio_live():
    global recording, transcript
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while recording:
            try:
                audio_data = recognizer.listen(source, phrase_time_limit=5)
                text = recognizer.recognize_google(audio_data)
                transcript += text + " "
                print("Live Transcription:", text)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print("Speech recognition error:", e)
                break

# --------------------------------------------------------------------------
# 10) DISEASE PREDICTION FUNCTION USING THE KERAS MODEL
# --------------------------------------------------------------------------
def predict_conditions(text):
    """
    Use the trained Keras model to predict a disease from the given transcript.
    The model expects vectorized (numeric) input, so we transform the raw text using the vectorizer.
    Then we map the predicted contiguous label back to the original label and finally to the disease name.
    """
    if keras_model is None:
        return "Model not loaded. Check logs."
    
    # Transform the raw text to vectorized representation.
    # Note: The vectorizer was adapted on training text.
    # The input must be a tensor; we wrap text in a list.
    text_vectorized = vectorizer(np.array([text]))
    
    predictions = keras_model.predict(text_vectorized)  # shape: (1, num_diseases)
    predicted_contiguous_label = int(np.argmax(predictions[0]))
    
    # Convert contiguous label to original label code
    predicted_original_label = index_to_label.get(predicted_contiguous_label, None)
    if predicted_original_label is None:
        return "Prediction label mapping error."
    
    # Use the reverse mapping (from mapping.json) to get disease name
    predicted_disease = reverse_mapping.get(predicted_original_label, "Unknown Condition")
    return predicted_disease

# --------------------------------------------------------------------------
# 11) FLASK ROUTES
# --------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, transcript
    transcript = ""
    recording = True
    thread = threading.Thread(target=transcribe_audio_live)
    thread.daemon = True
    thread.start()
    return jsonify({'message': 'Recording started'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, transcript
    recording = False
    key_info = extract_key_information(transcript)
    prediction = predict_conditions(transcript)
    return jsonify({
        'transcript': transcript,
        'key_info': key_info,
        'predictions': prediction
    })

# --------------------------------------------------------------------------
# 12) RUN THE APP
# --------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
