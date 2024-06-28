import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import librosa
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file1' not in request.files or 'file2' not in request.files:
        return 'No file part', 400

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        return 'No selected file', 400

    filename1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file1.filename))
    filename2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))

    file1.save(filename1)
    file2.save(filename2)

    similarity = calculate_similarity(filename1, filename2)
    return render_template('result.html', similarity=similarity)

def delayed_imports():
    global numba
    import numba

def calculate_similarity(file_path1, file_path2):
    delayed_imports()  # ここで numba をインポート
    y1, sr1 = librosa.load(file_path1, sr=22050, mono=True)  # デフォルトのサンプリングレートを使用し、モノラルに変換
    y2, sr2 = librosa.load(file_path2, sr=22050, mono=True)

    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

    mfcc1_mean = np.mean(mfcc1, axis=1)
    mfcc2_mean = np.mean(mfcc2, axis=1)

    similarity = np.dot(mfcc1_mean, mfcc2_mean) / (np.linalg.norm(mfcc1_mean) * np.linalg.norm(mfcc2_mean))
    return similarity * 100  # パーセント表記に変換


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
