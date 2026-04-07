"""Speech-to-text transcription with word-level timestamps using faster-whisper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float  # seconds
    end: float    # seconds


def transcribe(audio_path: str, model_size: str = "base") -> List[WordTiming]:
    """Transcribe audio file and return word-level timestamps.
    
    Args:
        audio_path: Path to audio file (mp3/wav).
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
    
    Returns:
        List of WordTiming with start/end times for each word.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    segments, _ = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
        language="en",
        task="translate",
    )
    
    words: List[WordTiming] = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append(WordTiming(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end,
                ))
    
    return words
