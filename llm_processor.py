from groq import Groq
from config import settings

client = Groq(api_key=settings.groq_api_key)
MODEL = "openai/gpt-oss-120b"

class LLMProcessor:
    def correct_transcript(self, raw_text: str, flagged_segments: list) -> str:
        flagged_info = ""
        if flagged_segments:
            flagged_info = "\n\nThese parts had low audio quality or possible gaps:\n"
            for seg in flagged_segments:
                flagged_info += (
                    f'- "{seg["text"]}" '
                    f'(around {seg["start"]:.1f}s-{seg["end"]:.1f}s)\n'
                )

        system_msg = (
            "You clean speech-to-text transcripts for a chatbot pipeline. "
            "You output ONLY the cleaned transcript text. "
            "No explanations, no headers, no 'becomes', no arrows, no showing before/after, "
            "no quotes around the text, no extra lines. Just the final cleaned sentence(s), nothing else."
        )

        user_msg = f"""Raw transcript:
{raw_text}
{flagged_info}
Fix unclear/garbled words using context. Fill small missing words if inferable.
Remove filler words (um, uh, like, matlab, actually) and stutters/repetitions (e.g. "how how many" -> "how many").
Do NOT summarize or change meaning. Do NOT answer the question. Do NOT add commentary.
Output only the cleaned transcript."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )

        result = response.choices[0].message.content.strip().strip('"')

        if "\n\n" in result or "becomes" in result.lower() or "->" in result:
            parts = [p.strip() for p in result.split("\n") if p.strip()]
            result = parts[-1] if parts else result

        return result
    
