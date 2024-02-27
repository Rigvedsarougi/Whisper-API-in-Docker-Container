from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import whisper
import torch
from typing import List

torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model("base", device=DEVICE)

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
async def handler(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided")

    # For each file, let's store the results in a list of dictionaries.
    results = []

    for file in files:
        # Create a temporary file.
        with NamedTemporaryFile(delete=True) as temp:
            # Write the user's uploaded file to the temporary file.
            with open(temp.name, "wb") as temp_file:
                temp_file.write(file.file.read())
            
            # Let's get the transcript of the temporary file.
            result = model.transcribe(temp.name)

            # Detect fraud in the transcript
            is_fraud, detected_keywords = detect_fraud(result['text'])

            # Now we can store the result object for this file.
            results.append({
                'filename': file.filename,
                'transcript': result['text'],
                'fraud_detected': is_fraud,
                'detected_keywords': detected_keywords
            })

    return JSONResponse(content={'results': results})


@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"
