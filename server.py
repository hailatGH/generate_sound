import os
import pickle
import numpy as np
import soundfile as sf
from flask import Flask, send_file, jsonify, url_for, request

from soundgenerator import SoundGenerator
from autoencoder import VAE
from train import SPECTROGRAMS_PATH
from generate import load_fsdd, select_spectrograms, save_signals

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
MIN_MAX_VALUES_PATH = "datasets/min_max_values.pkl"

app = Flask(__name__)


def generate_audio(num_spectrograms):
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    sampled_specs, sampled_min_max_values = select_spectrograms(specs,
                                                                file_paths,
                                                                min_max_values,
                                                                num_spectrograms)

    signals, _ = sound_generator.generate(sampled_specs,
                                          sampled_min_max_values)

    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs, sampled_min_max_values)

    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)


@app.route("/generate", methods=["POST"])
def generate():
    request_data = request.get_json()
    num_spectrograms = request_data['num_generated_audios']
    generate_audio(num_spectrograms)
    
    generated_audio_files = os.listdir(SAVE_DIR_GENERATED)
    original_audio_files = os.listdir(SAVE_DIR_ORIGINAL)
    
    audio_urls = {}
    generated_audio_urls = {}
    original_audio_urls = {}

    for file_name in generated_audio_files:
        file_url = url_for('get_single_generated_audio_file',
                           filename=file_name, _external=True)

        generated_audio_urls[file_name] = file_url
        
    for file_name in original_audio_files:
        file_url = url_for('get_single_original_audio_file',
                           filename=file_name, _external=True)

        original_audio_urls[file_name] = file_url
        
    audio_urls['generated'] = generated_audio_urls
    audio_urls['original'] = original_audio_urls

    return jsonify(audio_urls)


@app.route('/get_generated_audio_file/<filename>', methods=['GET'])
def get_single_generated_audio_file(filename):

    audio_file_path = os.path.join(SAVE_DIR_GENERATED, filename)
    if os.path.isfile(audio_file_path):
        return send_file(audio_file_path, mimetype='audio/wav')
    else:
        return 'Audio file not found', 404

@app.route('/get_original_audio_file/<filename>', methods=['GET'])
def get_single_original_audio_file(filename):

    audio_file_path = os.path.join(SAVE_DIR_ORIGINAL, filename)
    if os.path.isfile(audio_file_path):
        return send_file(audio_file_path, mimetype='audio/wav')
    else:
        return 'Audio file not found', 404
    

if __name__ == "__main__":
    app.run(debug=False)
