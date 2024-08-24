# `pip3 install assemblyai` (macOS)
# `pip install assemblyai` (Windows)

import assemblyai as aai
import yt_dlp

aai.settings.api_key = "c8b797cb78b64fa18d66c659d4378f42"
transcriber = aai.Transcriber()

# FILE_URL = "https://storage.googleapis.com/aai-web-samples/news.mp4"


def get_youtube_audio(url):
    with yt_dlp.YoutubeDL() as ydl:
        info = ydl.extract_info(url, download=False)

    for format in info["formats"][::-1]:
        if format["resolution"] == "audio only" and format["ext"] == "m4a":
            return format["url"]

    return None


YT_URL = "https://www.youtube.com/watch?v=Kcsn_6Wln60"

# audio_url = get_youtube_audio(YT_URL)
audio_url = "/Users/woutvossen/Downloads/wwe.mp3"

print("Audio URL:", audio_url)

if audio_url:
    print("Transcribing...")
    config = aai.TranscriptionConfig(speaker_labels=True)

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_url, config=config)
    print(transcript)
    transcript_text = ""
    for utterance in transcript.utterances:
        transcript_text += f"Speaker {utterance.speaker}: {utterance.text}\n"

    print(transcript_text)

    with open(
        "finetune_dataset/datasets/trump/raw_interviews/transcript.txt", "w"
    ) as file:
        file.write(transcript_text)

    print("Transcription completed and saved to transcript.txt")
