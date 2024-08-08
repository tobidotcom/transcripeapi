from flask import Flask, request, jsonify
import openai
import os
import tempfile
from pytube import YouTube

app = Flask(__name__)

# Set the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def download_youtube_audio(url):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    
    # Download the audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        audio_stream.download(output_path=temp_file.name)
        return temp_file.name

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({'error': 'No video URL provided'}), 400

    try:
        audio_file_path = download_youtube_audio(video_url)
        with open(audio_file_path, 'rb') as audio_file:
            transcription = openai.Audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return jsonify({'transcription': transcription['text']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
