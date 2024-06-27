from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'supersecretkey'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def get_mfcc(audio_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    if 'file1' not in request.files or 'file2' not in request.files:
        flash('両方のファイルを選択してください')
        return redirect(url_for('index'))
    
    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        flash('ファイルが選択されていません')
        return redirect(url_for('index'))
    
    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)
    
    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
    
    file1.save(filepath1)
    file2.save(filepath2)

    mfcc1 = get_mfcc(filepath1)
    mfcc2 = get_mfcc(filepath2)

    mfcc1_mean = np.mean(mfcc1, axis=1)
    mfcc2_mean = np.mean(mfcc2, axis=1)

    similarity = cosine_similarity([mfcc1_mean], [mfcc2_mean])[0][0] * 100

    return render_template('result.html', similarity=similarity)

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message=str(error)), 500

if __name__ == '__main__':
    app.run(debug=True)
