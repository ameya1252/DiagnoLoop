from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import threading
import openai
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY", "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")

# Global variables to manage recording state
recording = False
transcript = ""

def transcribe_audio_live():
    """
    Continuously listens to the microphone and updates the global `transcript`.
    """
    global recording, transcript
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        # Optionally, adjust for ambient noise:
        # recognizer.adjust_for_ambient_noise(source, duration=1)

        while recording:
            try:
                audio_data = recognizer.listen(source, phrase_time_limit=5)
                text = recognizer.recognize_google(audio_data)
                transcript += text + " "
                print("Live Transcription:", text)
            except sr.UnknownValueError:
                # Could not understand audio
                pass
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                break
            except Exception as e:
                print("Error transcribing live audio:", e)
                break

def extract_key_information(conversation_text):
    """
    Uses OpenAI GPT-3.5 (or higher) to extract key medical information.
    """
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
        key_info = response.choices[0].message.content.strip()
        return key_info
    except Exception as e:
        print("OpenAI API error:", e)
        return "Error extracting key information."

def predict_conditions(text):
    """
    Dummy placeholder for your condition-prediction logic.
    Replace this with your actual ML/model integration.
    """
    # Example: Return a list or string describing predicted conditions.
    return "No predictions implemented yet."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """
    Starts the audio recording in a separate thread.
    """
    global recording, transcript
    # Reset transcript each time we start recording (if desired)
    transcript = ""
    recording = True

    thread = threading.Thread(target=transcribe_audio_live)
    thread.daemon = True
    thread.start()

    return jsonify({'message': 'Recording started'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """
    Stops the recording, calls the key info extraction and prediction functions,
    and returns the results as JSON.
    """
    global recording, transcript
    recording = False  # Signal the transcribe thread to stop

    # Give the transcription thread a moment to exit gracefully (optional)
    # time.sleep(1)

    key_info = extract_key_information(transcript)
    predictions = predict_conditions(key_info)

    return jsonify({
        'transcript': transcript,
        'key_info': key_info,
        'predictions': predictions
    })

if __name__ == '__main__':
    app.run(debug=True)
