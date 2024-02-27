from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import whisper
import torch
from pydub import AudioSegment
from pydub.utils import mediainfo
from typing import List
import io
import requests

# Checking if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model:
model = whisper.load_model("base", device=DEVICE)

# Keywords for fraud detection
keywords = [
    'Global',
    'HANA',
    'Server',
    'Software'
]

app = FastAPI()

def detect_fraud(text):
    detected_keywords = [keyword for keyword in keywords if keyword in text]
    if detected_keywords:
        return True, detected_keywords
    else:
        return False, []

@app.post("/whisper/")
async def handler(file: UploadFile = UploadFile(...)):
    # Read the audio file
    audio_data = await file.read()

    # Convert audio to wav format
    audio = AudioSegment.from_file(io.BytesIO(audio_data))
    audio.export("temp.wav", format="wav")

    # Transcribe audio
    result = model.transcribe("temp.wav")

    # Detect fraud in the transcript
    is_fraud, detected_keywords = detect_fraud(result['text'])

    return JSONResponse(content={
        'transcript': result['text'],
        'fraud_detected': is_fraud,
        'detected_keywords': detected_keywords
    })

@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"
