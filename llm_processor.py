from groq import Groq
from config import settings

client = Groq(api_key=settings.groq_api_key)
MODEL = "llama-3.3-70b-versatile" 

class LLMProcessor:

    def correct_transcript(self, raw_text: str, flagged_segments: list) -> str:
       
        flagged_info = ""
        if flagged_segments:
            flagged_info = "\n\nNote: these parts had low audio quality or possible gaps:\n"
            for seg in flagged_segments:
                flagged_info += f"- \"{seg['text']}\" (around {seg['start']:.1f}s-{seg['end']:.1f}s)\n"

        prompt = f"""You are correcting a speech-to-text transcript that may have errors from background noise, low audio quality, or missing words due to gaps in the recording.

Raw transcript:
\"\"\"{raw_text}\"\"\"
{flagged_info}
Instructions:
- Fix unclear or garbled words using context from the surrounding sentence.
- Fill in small missing words/pieces naturally if they can be inferred.
- Do NOT add new information that isn't implied by context.
- Keep the corrected transcript close in length to the original do not summarize here.
- Return ONLY the corrected transcript text, nothing else.
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def summarize(self, corrected_text: str) -> str:
        
        prompt = f"""Summarize the following speech transcript into a short, clear, and natural summary. 
Do not copy sentences word-for-word  capture only the key points and intent in fewer words.

Transcript:
\"\"\"{corrected_text}\"\"\"

Return ONLY the summary text, nothing else.
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()