from flask import Flask, request, jsonify
import openai
import os
import tempfile
from pytube import YouTube, exceptions
from moviepy.editor import VideoFileClip

app = Flask(__name__)

# Set the OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def download_youtube_video(url):
    try:
        yt = YouTube(url)
        video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        
        if not video_stream:
            raise ValueError("No suitable video stream available for this video.")
        
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_stream.download(output_path=temp_video_file.name)
        temp_video_file.close()
        return temp_video_file.name
    except exceptions.PytubeError as e:
        app.logger.error(f"Pytube error: {e}")
    except ValueError as e:
        app.logger.error(f"Value error: {e}")
    except Exception as e:
        app.logger.error(f"Error downloading YouTube video: {e}")
    return None

def extract_audio_from_video(video_path):
    try:
        video = VideoFileClip(video_path)
        audio_path = video_path.replace('.mp4', '.wav')
        video.audio.write_audiofile(audio_path)
        video.close()
        return audio_path
    except Exception as e:
        app.logger.error(f"Error extracting audio from video: {e}")
    return None

@app.route('/transcribe', methods=['POST'])
def transcribe():
    video_file_path = None
    audio_file_path = None
    try:
        data = request.get_json()
        if not data or 'video_url' not in data:
            return jsonify({'error': 'Invalid input: video_url is required'}), 400

        video_url = data['video_url']
        video_file_path = download_youtube_video(video_url)

        if not video_file_path:
            return jsonify({'error': 'Failed to download video'}), 500

        audio_file_path = extract_audio_from_video(video_file_path)

        if not audio_file_path:
            return jsonify({'error': 'Failed to extract audio'}), 500

        with open(audio_file_path, 'rb') as audio_file:
            try:
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
        if video_file_path and os.path.exists(video_file_path):
            os.remove(video_file_path)
        if audio_file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

