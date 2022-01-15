import os
import sys
import numpy as np
import math
import librosa
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template


UPLOAD_FOLDER = './instance/htmlfi'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}


# Create Flask App
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(audio_file, track_duration=30):
    SAMPLE_RATE = 22050
    NUM_MFCC = 18
    N_FTT = 2048
    HOP_LENGTH = 512
    TRACK_DURATION = track_duration  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    NUM_SEGMENTS = 10

    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)

    for d in range(10):

        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(
            signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T

        return mfcc


# Upload files function
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.instance_path, 'htmlfi',
                      secure_filename(file.filename)))
            filename = f"./instance/htmlfi/{file.filename}"
            print(filename)
            return redirect(url_for('classify_and_show_results',
                                    filename=filename))
    return render_template("index.html")


# Classify and show results
@app.route('/results', methods=['GET'])
def classify_and_show_results():
    filename = request.args['filename']
    # Compute audio signal features
    # print(filename)
    new = extract_features(filename)

    X = new[np.newaxis, ..., np.newaxis]
    # Load model and perform inference
    model = load_model('my_model')
    predictions = model.predict(X)
    # Process predictions and render results
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    predicted_index = np.argmax(predictions)
    prediction = genres[int(predicted_index)]
    print(prediction)
    # Delete uploaded file
    # os.remove(filename)
    # Render results
    fileName = filename.split("/")[-1]
    return render_template("results.html",
                           filename=fileName,
                           pred=prediction)


if __name__ == "__main__":
    os.makedirs(os.path.join(app.instance_path, 'htmlfi'), exist_ok=True)
    app.debug = True
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
