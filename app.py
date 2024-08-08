from flask import Flask, request, jsonify
import openai
import os
import tempfile
from pytube import YouTube

app = Flask(__name__)

# Set the OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def download_youtube_audio(url):
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        # Create a temporary file to save the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        audio_stream.download(output_path=temp_file.name)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        app.logger.error(f"Error downloading YouTube audio: {e}")
        return None

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file_path = None
    try:
        data = request.get_json()
        if not data or 'video_url' not in data:
            return jsonify({'error': 'Invalid input: video_url is required'}), 400

        video_url = data['video_url']
        audio_file_path = download_youtube_audio(video_url)

        if not audio_file_path:
            return jsonify({'error': 'Failed to download audio'}), 500

        with open(audio_file_path, 'rb') as audio_file:
            try:
                # Call OpenAI API for transcription
                response = openai.Audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                return jsonify({'transcription': response['text']})
            except openai.error.OpenAIError as e:
                app.logger.error(f"OpenAI API error: {e}")
                return jsonify({'error': 'Failed to get transcription from OpenAI'}), 500

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500

    finally:
        if audio_file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

