import os
from flask import Flask, request, jsonify
from pytube import YouTube
from pydub import AudioSegment
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

def download_audio_from_youtube(video_url, output_path):
    yt = YouTube(video_url)
    stream = yt.streams.filter(only_audio=True).first()
    audio_file = stream.download(filename='downloaded_audio')
    audio = AudioSegment.from_file(audio_file)
    audio.export(output_path, format="mp3")
    os.remove(audio_file)
    return output_path

def transcribe_audio(file_path):
    with open(file_path, 'rb') as audio_file):
        transcript = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript['text']

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    video_url = data.get('video_url')
    output_path = 'audio.mp3'
    audio_file_path = download_audio_from_youtube(video_url, output_path)
    transcription = transcribe_audio(audio_file_path)
    os.remove(audio_file_path)
    return jsonify({'transcription': transcription})

if __name__ == "__main__":
    app.run()
