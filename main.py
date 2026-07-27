import shutil
import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException

from stt_engine import STTEngine
from llm_processor import LLMProcessor

app = FastAPI(title="Speech-to-Text API with Auto-Correction & Summarization")

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


        corrected_text = llm_processor.correct_transcript(
            stt_result["raw_text"],
            stt_result["flagged_segments"]
        )

        summary = llm_processor.summarize(corrected_text)

        return {
            "language": stt_result["language"],
            "raw_transcript": stt_result["raw_text"],
            "corrected_transcript": corrected_text,
            "short_summary": summary,
            "flagged_segments_count": len(stt_result["flagged_segments"])
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


