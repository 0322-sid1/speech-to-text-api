import shutil
import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException

from stt_engine import STTEngine
from llm_processor import LLMProcessor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Speech-to-Text API with Auto-Correction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

stt_engine = STTEngine()
llm_processor = LLMProcessor()

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        stt_result = stt_engine.transcribe(temp_path)

        if not stt_result["raw_text"]:
            raise HTTPException(status_code=400, detail="No speech detected in audio.")

        clean_text = llm_processor.correct_transcript(
            stt_result["raw_text"],
            stt_result["flagged_segments"]
        )

        return {
            "language": stt_result["language"],
            "raw_transcript": stt_result["raw_text"],
            "clean_transcript": clean_text,
            "flagged_segments_count": len(stt_result["flagged_segments"])
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
