from faster_whisper import WhisperModel
from config import settings
import tempfile
import os


class STTEngine:
    def __init__(self):
        self.model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            cpu_threads=4,       
            num_workers=1 
        )

    def transcribe(self, audio_path: str) -> dict:
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True
        )

        full_text = ""
        flagged_segments = []
        segment_list = list(segments) 

        GAP_THRESHOLD = 0.4

        for i, seg in enumerate(segment_list):
            full_text += seg.text + " "

            if seg.avg_logprob < -0.7 or seg.no_speech_prob > 0.4:
                flagged_segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "reason": "low_confidence"
                })

            if i > 0:
                prev_end = segment_list[i - 1].end
                gap = seg.start - prev_end
                if gap > GAP_THRESHOLD:
                    flagged_segments.append({
                        "start": prev_end,
                        "end": seg.start,
                        "text": "[possible missing audio]",
                        "reason": f"silence_gap_{gap:.2f}s"
                    })

        return {
            "raw_text": full_text.strip(),
            "language": info.language,
            "flagged_segments": flagged_segments
        }
    
           
    
