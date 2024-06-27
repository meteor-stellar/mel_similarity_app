import os
from flask import Flask, render_template, request, redirect, url_for, flash
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'Applerin1101'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            filename1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file1.filename))
            filename2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))
            file1.save(filename1)
            file2.save(filename2)
            
            similarity = calculate_similarity(filename1, filename2)
            return redirect(url_for('result', similarity=similarity))
        else:
            flash('ファイルが選択されていないか、サポートされていない形式です。再度選択してください。')
            return redirect(url_for('index'))

@app.route('/result')
def result():
    similarity = request.args.get('similarity', type=float)
    return render_template('result.html', similarity=similarity)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_similarity(file_path1, file_path2):
    y1, sr1 = librosa.load(file_path1)
    y2, sr2 = librosa.load(file_path2)

    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

    mfcc1_mean = np.mean(mfcc1, axis=1)
    mfcc2_mean = np.mean(mfcc2, axis=1)

    similarity = cosine_similarity([mfcc1_mean], [mfcc2_mean])[0][0] * 100
    return similarity

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
