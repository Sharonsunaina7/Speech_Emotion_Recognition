import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import wavio
import numpy as np
import librosa
from keras.models import load_model
from langdetect import detect
import speech_recognition as sr

gender_model = load_model('gsr_model.h5')
emotion_model = load_model('ser_model.h5')

def extract_features(file_path, n_mfcc=40):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return np.array([mfccs_mean])

def predict_gender(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=-1)  
    features = np.expand_dims(features, axis=0)   
    prediction = gender_model.predict(features)
    return 'Male' if np.argmax(prediction) == 1 else 'Female'

def predict_emotion(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=-1) 
    features = np.expand_dims(features, axis=0)   
    prediction = emotion_model.predict(features)
    emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprise']
    return emotions[np.argmax(prediction)]

def is_english(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        language = detect(text)
        return language == 'en'
    except sr.UnknownValueError:
        return False
    except sr.RequestError:
        return False

def record_audio():
    duration = 5
    fs = 44100  
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)  
    sd.wait()
    print("Recording complete")
    wavio.write("recorded.wav", recording, fs, sampwidth=2)
    return "recorded.wav"

class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")
        self.root.geometry("500x400")
        self.root.configure(bg="lightgrey")

        self.title_label = tk.Label(root, text="Emotion Detector", font=("Arial", 24), bg="lightgrey")
        self.title_label.pack(pady=20)

        self.record_button = tk.Button(root, text="Record Audio", command=self.record_audio, font=("Arial", 12), bg="darkblue", fg="white", width=20)
        self.record_button.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Audio", command=self.upload_audio, font=("Arial", 12), bg="darkblue", fg="white", width=20)
        self.upload_button.pack(pady=10)

        self.gender_label = tk.Label(root, text="Gender: ", font=("Arial", 14), bg="lightgrey")
        self.gender_label.pack(pady=5)

        self.emotion_label = tk.Label(root, text="Emotion: ", font=("Arial", 14), bg="lightgrey")
        self.emotion_label.pack(pady=5)

    def record_audio(self):
        file_path = record_audio()
        self.predict(file_path)

    def upload_audio(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.predict(file_path)

    def predict(self, file_path):
        if not is_english(file_path):
            messagebox.showerror("Error", "Only English audio is accepted.")
            self.gender_label.config(text="Gender: N/A")
            self.emotion_label.config(text="Emotion: N/A")
            return
        
        gender = predict_gender(file_path)
        self.gender_label.config(text=f"Gender: {gender}")

        if gender == 'Male':
            messagebox.showerror("Error", "Only female voices are accepted.")
            self.emotion_label.config(text="Emotion: N/A")
        else:
            emotion = predict_emotion(file_path)
            self.emotion_label.config(text=f"Emotion: {emotion}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognitionApp(root)
    root.mainloop()
