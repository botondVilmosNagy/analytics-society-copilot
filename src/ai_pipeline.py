from __future__ import annotations

import io
import os
from typing import List, Tuple

from openai import OpenAI

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .rag import RetrievalChunk, format_context


class VoicePipeline:
    def __init__(self) -> None:
        self.client = OpenAI()
        self.whisper_model = os.getenv("WHISPER_MODEL", "whisper-1")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.tts_model = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
        self.tts_voice = os.getenv("TTS_VOICE", "alloy")

    def transcribe_audio_bytes(self, audio_bytes: bytes) -> str:
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.name = "input.wav"

        transcript = self.client.audio.transcriptions.create(
            model=self.whisper_model,
            file=audio_buffer,
        )
        return transcript.text.strip()

    def generate_answer(self, question: str, chunks: List[RetrievalChunk], mode: str) -> Tuple[str, float]:
        context_block = format_context(chunks)
        user_prompt = build_user_prompt(question, context_block, mode)

        completion = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        answer = completion.choices[0].message.content or "I could not generate a response."
        confidence = self._confidence_from_chunks(chunks)
        return answer.strip(), confidence

    def synthesize_speech(self, text: str) -> bytes:
        response = self.client.audio.speech.create(
            model=self.tts_model,
            voice=self.tts_voice,
            input=text,
        )
        return response.content

    @staticmethod
    def _confidence_from_chunks(chunks: List[RetrievalChunk]) -> float:
        if not chunks:
            return 0.0
        return sum(chunk.score for chunk in chunks) / len(chunks)
