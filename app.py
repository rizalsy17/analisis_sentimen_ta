import time
from flask import Flask, flash, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
# import numpy as np
import pickle
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload
import google.oauth2.credentials
import json
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import io
# import re
import os
from werkzeug.utils import secure_filename
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import re
import uuid
import tempfile

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_COOKIE_NAME'] = 'google_session'
app.config['TEMP_FOLDER'] = os.path.join(tempfile.gettempdir(), 'analisis_sentimen_temp')

if not os.path.exists(app.config['TEMP_FOLDER']):
    os.makedirs(app.config['TEMP_FOLDER'])

@app.template_filter('escapejs')
def escapejs_filter(s):
    """Escape JSON for use in JavaScript."""
    if isinstance(s, str):
        s = s.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
        s = s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    return s

CLIENT_SECRETS_FILE = "client_secret.json" 
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive.file']

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/login')
def login():
    # Proses OAuth untuk mendapatkan token akses
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES)
    flow.redirect_uri = url_for('oauth2callback', _external=True)
    
    authorization_url, state = flow.authorization_url(
        access_type='offline', include_granted_scopes='true')
    
    # Simpan state dalam session untuk digunakan saat callback
    session['state'] = state
    
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    # Menangani callback dari Google dan mendapatkan token akses
    state = session['state']
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
    flow.redirect_uri = url_for('oauth2callback', _external=True)
    
    authorization_response = request.url
    flow.fetch_token(authorization_response=authorization_response)
    
    credentials = flow.credentials
    session['credentials'] = credentials_to_dict(credentials)
    
    return redirect(url_for('upload_file'))

@app.route('/drive')
def drive():
    # Menggunakan token untuk mengakses Google Drive API
    credentials = session.get('credentials')
    if not credentials:
        return redirect(url_for('login'))
    
    # Bangun service API
    drive_service = googleapiclient.discovery.build(API_NAME, API_VERSION, credentials=credentials)
    
    # Akses data Google Drive
    results = drive_service.files().list(pageSize=10, fields="nextPageToken, files(id, name)").execute()
    files = results.get('files', [])
    
    if not files:
        return 'No files found.'
    
    file_list = '<br>'.join([f'{file["name"]} ({file["id"]})' for file in files])
    
    return f'Files:<br>{file_list}'

@app.route('/preview-data', methods=['POST'])
def handle_preview_post():
    if not request.form:
        flash("Tidak ada file yang dikirim", "error")
        return redirect(url_for('index'))
    
    # 1. Kumpulkan data dari form
    selected_folders = {}
    
    # Periksa apakah data dikirim sebagai JSON dalam hidden input
    if 'selectedFoldersInput' in request.form and request.form['selectedFoldersInput']:
        try:
            selected_folders = json.loads(request.form['selectedFoldersInput'])
        except:
            pass
    
    # Jika tidak ada JSON, coba ambil dari input terpisah
    if not selected_folders:
        for key in request.form:
            if key.startswith("folders["):
                # Contoh key: folders[fasilitas kelas][0][name]
                parts = key.replace("folders[", "").replace("]", "").split("[")
                if len(parts) == 3:
                    folder_name, index, field = parts
                    index = int(index)
                    
                    if folder_name not in selected_folders:
                        selected_folders[folder_name] = []
                    
                    # Tambahkan slot jika belum ada
                    while len(selected_folders[folder_name]) <= index:
                        selected_folders[folder_name].append({})
                    
                    selected_folders[folder_name][index][field] = request.form[key]
    
    if not selected_folders:
        flash("Tidak ada file yang dipilih", "warning")
        return redirect(url_for('index'))
    
    # 2. Ambil access token dari session atau request
    access_token = session.get('access_token')
    
    if not access_token:
        flash("Akses Google Drive tidak tersedia", "error")
        return redirect(url_for('index'))
    
    # Buat kredensial dari access token
    credentials = google.oauth2.credentials.Credentials(access_token) 
    drive_service = googleapiclient.discovery.build('drive', 'v3', credentials=credentials)
    
    all_data = {}
    
    # 3. Proses file per folder
    for folder_name, files in selected_folders.items():
        merged_df = pd.DataFrame()
        
        for file in files:
            file_id = file.get('id')
            file_name = file.get('name')
            mime_type = file.get('mimeType')
            
            try:
                if mime_type == 'application/vnd.google-apps.spreadsheet':
                    export_mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    request_drive = drive_service.files().export_media(fileId=file_id, mimeType=export_mime_type)
                else:
                    request_drive = drive_service.files().get_media(fileId=file_id)
                
                fh = io.BytesIO()
                downloader = googleapiclient.http.MediaIoBaseDownload(fh, request_drive)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                fh.seek(0)
                
                # Baca file ke dataframe
                if file_name.lower().endswith('.csv'):
                    df = pd.read_csv(fh)
                else:
                    df = pd.read_excel(fh, engine='openpyxl')
                
                # Filter kolom berdasarkan nama folder
                folder_lower = folder_name.strip().lower()
                
                # Untuk folder fasilitas kelas
                if folder_lower == 'fasilitas kelas':
                    # Cek apakah kolom yang diperlukan ada
                    required_cols = ['Ruang Kelas', 'Kritik dan Saran']
                    
                    # Jika nama kolom sedikit berbeda, coba cari kolom yang mirip
                    if not all(col in df.columns for col in required_cols):
                        for req_col in required_cols:
                            if req_col not in df.columns:
                                # Cari kolom yang mirip
                                for col in df.columns:
                                    if req_col.lower() in col.lower():
                                        # Rename kolom ke nama yang diharapkan
                                        df = df.rename(columns={col: req_col})
                    
                    # Filter kolom jika semua kolom yang diperlukan ada
                    if all(col in df.columns for col in required_cols):
                        df = df[required_cols]
                    else:
                        app.logger.warning(f"Kolom yang diperlukan tidak ditemukan di file {file_name}")
                        continue
                
                # Untuk folder pembelajaran dosen
                elif folder_lower == 'pembelajaran dosen':
                    required_cols = ['Nama Dosen Pengampu', 'Kritik dan Saran']
                    
                    # Jika nama kolom sedikit berbeda, coba cari kolom yang mirip
                    if not all(col in df.columns for col in required_cols):
                        for req_col in required_cols:
                            if req_col not in df.columns:
                                # Cari kolom yang mirip
                                for col in df.columns:
                                    if req_col.lower() in col.lower():
                                        # Rename kolom ke nama yang diharapkan
                                        df = df.rename(columns={col: req_col})
                    
                    # Filter kolom jika semua kolom yang diperlukan ada
                    if all(col in df.columns for col in required_cols):
                        df = df[required_cols]
                    else:
                        app.logger.warning(f"Kolom yang diperlukan tidak ditemukan di file {file_name}")
                        continue
                
                # Gabungkan dataframe
                merged_df = pd.concat([merged_df, df], ignore_index=True)
                
            except Exception as e:
                app.logger.error(f"Error reading file {file_name}: {str(e)}")
                continue
        
        # Simpan data hasil gabungan ke dalam all_data
        if not merged_df.empty:
            all_data[folder_name] = merged_df.to_dict(orient='records')
    
    # 4. Generate a unique ID for this data
    data_id = str(uuid.uuid4())
    
    # 5. Save the data to a temporary file instead of session
    temp_file_path = os.path.join(app.config['TEMP_FOLDER'], f"{data_id}.json")
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f)
    
    # 6. Store only the data ID in session (much smaller)
    session['preview_data_id'] = data_id
    
    return render_template('preview.html', preview_data=all_data)

@app.route('/save-token', methods=['POST'])
def save_token():
    try:
        data = request.json
        access_token = data.get('access_token')
        
        if access_token:
            session['access_token'] = access_token
            return jsonify({'success': True, 'message': 'Token tersimpan'})
        else:
            return jsonify({'success': False, 'message': 'Access token tidak ditemukan'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    
@app.route('/preview')
def preview():
    # Get the data ID from session
    data_id = session.get('preview_data_id')
    
    if not data_id:
        flash("Tidak ada data preview yang tersedia", "warning")
        return redirect(url_for('index'))
    
    # Load the data from temporary file
    temp_file_path = os.path.join(app.config['TEMP_FOLDER'], f"{data_id}.json")
    
    try:
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            preview_data = json.load(f)
    except FileNotFoundError:
        flash("Data preview tidak ditemukan atau sudah kedaluwarsa", "error")
        return redirect(url_for('index'))
    
    return render_template('preview.html', preview_data=preview_data)

def credentials_to_dict(credentials):
    """Convert the credentials to a dictionary."""
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

# Add a cleanup route to remove old temporary files (optional)
@app.route('/cleanup-temp', methods=['POST'])
def cleanup_temp():
    # Admin function to clean up old temp files
    # Add authentication here in production
    
    count = 0
    for filename in os.listdir(app.config['TEMP_FOLDER']):
        if filename.endswith('.json'):
            file_path = os.path.join(app.config['TEMP_FOLDER'], filename)
            # Check if file is older than 24 hours
            if os.path.getmtime(file_path) < time.time() - 86400:
                os.remove(file_path)
                count += 1
    
    return jsonify({"success": True, "files_removed": count})

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj

def generate_label(text, sentiment_words):
    """
    Generate a simulated label for a text based on sentiment words
    """
    words = text.split()
    
    # Count matches for each sentiment
    positif_count = sum(1 for word in words if word in sentiment_words.get('positif', []))
    netral_count = sum(1 for word in words if word in sentiment_words.get('netral', []))
    negatif_count = sum(1 for word in words if word in sentiment_words.get('negatif', []))
    
    # Add some randomness to simulate human labeling
    positif_count += np.random.randint(0, 2)
    netral_count += np.random.randint(0, 2)
    negatif_count += np.random.randint(0, 2)
    
    # Get the sentiment with highest count
    counts = {'positif': positif_count, 'netral': netral_count, 'negatif': negatif_count}
    max_sentiment = max(counts, key=counts.get)
    
    return max_sentiment

def calculate_metrics(true_labels, predictions):
    
    if not true_labels or not predictions:
        # Return empty metrics if no data
        empty_cm = {
            'positif': {'positif': 0, 'netral': 0, 'negatif': 0},
            'netral': {'positif': 0, 'netral': 0, 'negatif': 0},
            'negatif': {'positif': 0, 'netral': 0, 'negatif': 0}
        }
        empty_metrics = {
            'accuracy': 0,
            'positif': {'precision': 0, 'recall': 0, 'f1': 0, 'specificity': 0},
            'netral': {'precision': 0, 'recall': 0, 'f1': 0, 'specificity': 0},
            'negatif': {'precision': 0, 'recall': 0, 'f1': 0, 'specificity': 0},
            'macro': {'precision': 0, 'recall': 0, 'f1': 0, 'specificity': 0}
        }
        return empty_cm, empty_metrics
    
    # Define the classes
    classes = ['positif', 'netral', 'negatif']
    
    # Create confusion matrix
    cm_array = confusion_matrix(true_labels, predictions, labels=classes)
    
    # Convert to dictionary format for easier template access
    cm = {
        'positif': {'positif': cm_array[0][0], 'netral': cm_array[0][1], 'negatif': cm_array[0][2]},
        'netral': {'positif': cm_array[1][0], 'netral': cm_array[1][1], 'negatif': cm_array[1][2]},
        'negatif': {'positif': cm_array[2][0], 'netral': cm_array[2][1], 'negatif': cm_array[2][2]}
    }
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    # Calculate precision, recall, and f1 for each class
    precision = precision_score(true_labels, predictions, labels=classes, average=None, zero_division=0)
    recall = recall_score(true_labels, predictions, labels=classes, average=None, zero_division=0)
    f1 = f1_score(true_labels, predictions, labels=classes, average=None, zero_division=0)
    
    # Calculate specificity for each class
    specificity = []
    for i, class_name in enumerate(classes):
        # Create binary classification for current class
        true_binary = [1 if label == class_name else 0 for label in true_labels]
        pred_binary = [1 if label == class_name else 0 for label in predictions]
        
        # Calculate TN and FP
        tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
        
        # Calculate specificity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'positif': {
            'precision': precision[0],
            'recall': recall[0],
            'f1': f1[0],
            'specificity': specificity[0]
        },
        'netral': {
            'precision': precision[1],
            'recall': recall[1],
            'f1': f1[1],
            'specificity': specificity[1]
        },
        'negatif': {
            'precision': precision[2],
            'recall': recall[2],
            'f1': f1[2],
            'specificity': specificity[2]
        },
        'macro': {
            'precision': sum(precision) / 3,
            'recall': sum(recall) / 3,
            'f1': sum(f1) / 3,
            'specificity': sum(specificity) / 3
        }
    }
    
    return cm, metrics

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    
    # Download necessary NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk_available = True
    except:
        nltk_available = False
except ImportError:
    nltk_available = False

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj

def preprocess_text(text, stemming=True):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    if nltk_available:
        try:
            # Tokenize
            words = nltk.word_tokenize(text)
            
            # Remove stopwords (Indonesian and English)
            stop_words = set(stopwords.words('english'))
            # Add Indonesian stopwords
            indo_stopwords = ['yang', 'dan', 'di', 'dengan', 'untuk', 'tidak', 'ini', 'dari', 'dalam', 'akan', 'pada', 'juga', 'saya', 'ke', 'bisa', 'ada', 'itu', 'adalah']
            stop_words.update(indo_stopwords)
            words = [word for word in words if word not in stop_words]
            
            # Apply stemming if requested
            if stemming:
                stemmer = PorterStemmer()
                words = [stemmer.stem(word) for word in words]
            
            # Join words back into a string
            return ' '.join(words)
        except Exception as e:
            # If any NLTK processing fails, fall back to simple preprocessing
            print(f"NLTK processing failed: {str(e)}. Using simple preprocessing.")
            return text
    else:
        # Simple tokenization by splitting on whitespace
        words = text.split()
        return ' '.join(words)

def generate_label(text, sentiment_words):
    """
    Generate a more accurate label for a text based on sentiment words
    with weighted scoring
    """
    # Simple preprocessing for matching
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    # Initialize scores with weights
    scores = {
        'positif': 0,
        'netral': 0,
        'negatif': 0
    }
    
    # Count matches for each sentiment with weights
    for word in words:
        if word in sentiment_words.get('positif', []):
            scores['positif'] += 1.5  # Higher weight for positive words
        if word in sentiment_words.get('netral', []):
            scores['netral'] += 1.0
        if word in sentiment_words.get('negatif', []):
            scores['negatif'] += 2.0  # Higher weight for negative words
    
    # Get the sentiment with highest score
    if max(scores.values()) == 0:
        # If no sentiment words found, default to neutral
        return 'netral'
    else:
        return max(scores, key=scores.get)

def calculate_metrics(true_labels, predictions):
    if not true_labels or not predictions:
        # Return empty metrics if no data
        empty_cm = {
            'positif': {'positif': 0, 'netral': 0, 'negatif': 0},
            'netral': {'positif': 0, 'netral': 0, 'negatif': 0},
            'negatif': {'positif': 0, 'netral': 0, 'negatif': 0}
        }
        empty_metrics = {
            'accuracy': 0,
            'positif': {'precision': 0, 'recall': 0, 'f1': 0, 'specificity': 0},
            'netral': {'precision': 0, 'recall': 0, 'f1': 0, 'specificity': 0},
            'negatif': {'precision': 0, 'recall': 0, 'f1': 0, 'specificity': 0},
            'macro': {'precision': 0, 'recall': 0, 'f1': 0, 'specificity': 0}
        }
        return empty_cm, empty_metrics
    
    # Define the classes
    classes = ['positif', 'netral', 'negatif']
    
    # Create confusion matrix
    cm_array = confusion_matrix(true_labels, predictions, labels=classes)
    
    # Generate confusion matrix visualization
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the visualization
        os.makedirs('static/images', exist_ok=True)
        cm_plot_path = os.path.join('static', 'images', 'confusion_matrix.png')
        plt.savefig(cm_plot_path)
        plt.close()
    except Exception as e:
        print(f"Error generating confusion matrix visualization: {str(e)}")
    
    # Convert to dictionary format for easier template access
    cm = {
        'positif': {'positif': cm_array[0][0], 'netral': cm_array[0][1], 'negatif': cm_array[0][2]},
        'netral': {'positif': cm_array[1][0], 'netral': cm_array[1][1], 'negatif': cm_array[1][2]},
        'negatif': {'positif': cm_array[2][0], 'netral': cm_array[2][1], 'negatif': cm_array[2][2]}
    }
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    # Calculate precision, recall, and f1 for each class
    precision = precision_score(true_labels, predictions, labels=classes, average=None, zero_division=0)
    recall = recall_score(true_labels, predictions, labels=classes, average=None, zero_division=0)
    f1 = f1_score(true_labels, predictions, labels=classes, average=None, zero_division=0)
    
    # Calculate specificity for each class
    specificity = []
    for i, class_name in enumerate(classes):
        # Create binary classification for current class
        true_binary = [1 if label == class_name else 0 for label in true_labels]
        pred_binary = [1 if label == class_name else 0 for label in predictions]
        
        # Calculate TN and FP
        tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
        
        # Calculate specificity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'positif': {
            'precision': precision[0],
            'recall': recall[0],
            'f1': f1[0],
            'specificity': specificity[0]
        },
        'netral': {
            'precision': precision[1],
            'recall': recall[1],
            'f1': f1[1],
            'specificity': specificity[1]
        },
        'negatif': {
            'precision': precision[2],
            'recall': recall[2],
            'f1': f1[2],
            'specificity': specificity[2]
        },
        'macro': {
            'precision': sum(precision) / 3,
            'recall': sum(recall) / 3,
            'f1': sum(f1) / 3,
            'specificity': sum(specificity) / 3
        }
    }
    
    # Generate metrics visualization
    try:
        plt.figure(figsize=(10, 6))
        metrics_data = {
            'Precision': [metrics['positif']['precision'], metrics['netral']['precision'], metrics['negatif']['precision']],
            'Recall': [metrics['positif']['recall'], metrics['netral']['recall'], metrics['negatif']['recall']],
            'F1 Score': [metrics['positif']['f1'], metrics['netral']['f1'], metrics['negatif']['f1']],
            'Specificity': [metrics['positif']['specificity'], metrics['netral']['specificity'], metrics['negatif']['specificity']]
        }
        
        # Plot metrics
        bar_width = 0.2
        x = np.arange(3)  # 3 classes
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            plt.bar(x + i * bar_width, values, width=bar_width, label=metric)
        
        plt.xlabel('Sentiment Class')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Class')
        plt.xticks(x + bar_width * 1.5, classes)
        plt.legend()
        plt.tight_layout()
        
        # Save the visualization
        metrics_plot_path = os.path.join('static', 'images', 'metrics_comparison.png')
        plt.savefig(metrics_plot_path)
        plt.close()
    except Exception as e:
        print(f"Error generating metrics visualization: {str(e)}")
    
    return cm, metrics

def train_test_split(texts, labels, indices, group_keys, test_size=0.2, random_state=None):
    """
    Custom train_test_split function that also splits the group_keys
    """
    from sklearn.model_selection import train_test_split as sklearn_split
    
    # Use stratify if possible
    try:
        X_train, X_test, y_train, y_test, train_indices, test_indices = sklearn_split(
            texts, labels, indices, test_size=test_size, random_state=random_state, stratify=labels
        )
    except ValueError:
        # If stratification fails (e.g., too few samples), do without stratification
        X_train, X_test, y_train, y_test, train_indices, test_indices = sklearn_split(
            texts, labels, indices, test_size=test_size, random_state=random_state
        )
    
    # Also split the group_keys
    group_train = [group_keys[i] for i in train_indices]
    group_test = [group_keys[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices, group_train, group_test

def analyze_sentiment(app):
    # 1. Get the data ID from session
    data_id = session.get('preview_data_id')
    if not data_id:
        flash("Tidak ada data untuk dianalisis", "error")
        return redirect(url_for('index'))
    
    # 2. Load the data from temporary file
    temp_file_path = os.path.join(app.config['TEMP_FOLDER'], f"{data_id}.json")
    if not os.path.exists(temp_file_path):
        flash("Data tidak ditemukan", "error")
        return redirect(url_for('index'))
    
    with open(temp_file_path, 'r', encoding='utf-8') as f:
        preview_data = json.load(f)
    
    # 3. Load sentiment keywords
    try:
        with open('sentimen_words.json', 'r', encoding='utf-8') as f:
            sentiment_words = json.load(f)
    except Exception as e:
        app.logger.error(f"Error loading sentiment words: {str(e)}")
        flash("Gagal memuat kata kunci sentimen", "error")
        return redirect(url_for('preview_data'))
    
    # 4. Process sentiment analysis with train-test split for each folder
    sentiment_results = {}
    train_results = {}
    test_results = {}
    all_predictions = []
    all_true_labels = []
    
    # New dictionaries to store grouped results
    grouped_results = {}
    grouped_summary = {}
    
    # Create directory for visualizations if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    for folder_name, folder_data in preview_data.items():
        # Skip if folder is empty
        if not folder_data:
            continue
            
        # Initialize results for this folder
        sentiment_results[folder_name] = []
        train_results[folder_name] = []
        test_results[folder_name] = []
        grouped_results[folder_name] = {}
        
        # Prepare data for analysis
        texts = []
        items = []
        group_keys = []  # Store the grouping key (professor name or room name)
        
        for item in folder_data:
            # Get the text to analyze based on folder type
            if folder_name.lower() == 'fasilitas kelas':
                text_to_analyze = item.get('Kritik dan Saran', '')
                # Get the room name for grouping
                group_key = item.get('Ruang Kelas', 'Unknown')
            elif folder_name.lower() == 'pembelajaran dosen':
                text_to_analyze = item.get('Kritik dan Saran', '')
                # Get the professor name for grouping
                group_key = item.get('Nama Dosen Pengampu', 'Unknown')
            else:
                continue
                
            if text_to_analyze and isinstance(text_to_analyze, str):  # Only include valid text items
                texts.append(text_to_analyze)
                items.append(item)
                group_keys.append(group_key)
        
        # Skip if no valid texts found
        if not texts:
            continue
            
        # Preprocess texts for vectorization
        preprocessed_texts = []
        for text in texts:
            preprocessed_texts.append(preprocess_text(text))
        
        # Create labels based on sentiment words
        labels = []
        for text in texts:  # Use original texts for better matching
            # Generate label based on sentiment words
            label = generate_label(text, sentiment_words)
            labels.append(label)
        

        X_train, X_test, y_train, y_test, train_indices, test_indices, group_train, group_test = train_test_split(
            preprocessed_texts, 
            labels,
            range(len(texts)),
            group_keys,
            test_size=0.2, 
            random_state=42
        )
        
        # Create TF-IDF vectorizer and transform texts to feature vectors
        try:
            vectorizer = TfidfVectorizer(
                min_df=1,  # Minimum document frequency
                max_df=0.9,  # Maximum document frequency
                ngram_range=(1, 2),  # Use unigrams and bigrams
                sublinear_tf=True  # Apply sublinear tf scaling
            )
            
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
        except Exception as e:
            app.logger.error(f"Vectorization error: {str(e)}")
            # Fall back to CountVectorizer if TF-IDF fails
            try:
                vectorizer = CountVectorizer(min_df=1)
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
            except Exception as e2:
                app.logger.error(f"CountVectorizer also failed: {str(e2)}")
                flash(f"Error during text processing in folder {folder_name}", "error")
                continue
        
        # Train MultinomialNB classifier with optimized alpha
        try:
            classifier = MultinomialNB(alpha=0.1)  # Lower alpha for less smoothing
            classifier.fit(X_train_vec, y_train)
        except Exception as e:
            app.logger.error(f"Classifier training error: {str(e)}")
            # Try with default alpha if optimized fails
            try:
                classifier = MultinomialNB()
                classifier.fit(X_train_vec, y_train)
            except Exception as e2:
                app.logger.error(f"Classifier training with default alpha also failed: {str(e2)}")
                flash(f"Error during model training in folder {folder_name}", "error")
                continue
        
        # Process training data
        train_items_indices = list(train_indices)
        for i, idx in enumerate(train_items_indices):
            result = items[idx].copy()
            text = texts[idx]
            group_key = group_keys[idx]
            
            # Get model predictions
            text_processed = preprocess_text(text)
            text_vec = vectorizer.transform([text_processed])
            
            # Get predicted sentiment and probabilities
            sentiment = classifier.predict(text_vec)[0]
            probabilities = classifier.predict_proba(text_vec)[0]
            
            # Map probabilities to sentiment classes
            prob_dict = {
                'positif': probabilities[list(classifier.classes_).index('positif')] if 'positif' in classifier.classes_ else 0,
                'netral': probabilities[list(classifier.classes_).index('netral')] if 'netral' in classifier.classes_ else 0,
                'negatif': probabilities[list(classifier.classes_).index('negatif')] if 'negatif' in classifier.classes_ else 0
            }
            
            # Add sentiment results to the item
            result['sentiment'] = sentiment
            result['probabilities'] = prob_dict
            result['dataset'] = 'training'
            result['true_label'] = y_train[i]
            
            # Add to grouped results
            if group_key not in grouped_results[folder_name]:
                grouped_results[folder_name][group_key] = []
            grouped_results[folder_name][group_key].append(result)
            
            train_results[folder_name].append(result)
            sentiment_results[folder_name].append(result)
        
        # Process testing data
        folder_test_predictions = []
        folder_test_true_labels = []
        
        test_items_indices = list(test_indices)
        for i, idx in enumerate(test_items_indices):
            result = items[idx].copy()
            text = texts[idx]
            group_key = group_keys[idx]
            
            # Get model predictions
            text_processed = preprocess_text(text)
            text_vec = vectorizer.transform([text_processed])
            
            # Get predicted sentiment and probabilities
            sentiment = classifier.predict(text_vec)[0]
            probabilities = classifier.predict_proba(text_vec)[0]
            
            # Map probabilities to sentiment classes
            prob_dict = {
                'positif': probabilities[list(classifier.classes_).index('positif')] if 'positif' in classifier.classes_ else 0,
                'netral': probabilities[list(classifier.classes_).index('netral')] if 'netral' in classifier.classes_ else 0,
                'negatif': probabilities[list(classifier.classes_).index('negatif')] if 'negatif' in classifier.classes_ else 0
            }
            
            # Add sentiment results to the item
            result['sentiment'] = sentiment
            result['probabilities'] = prob_dict
            result['dataset'] = 'testing'
            result['true_label'] = y_test[i]
            
            # Add to grouped results
            if group_key not in grouped_results[folder_name]:
                grouped_results[folder_name][group_key] = []
            grouped_results[folder_name][group_key].append(result)
            
            folder_test_predictions.append(sentiment)
            folder_test_true_labels.append(y_test[i])
            all_predictions.append(sentiment)
            all_true_labels.append(y_test[i])
            
            test_results[folder_name].append(result)
            sentiment_results[folder_name].append(result)
        
        # Generate word frequency visualization for each sentiment class
        try:
            for sentiment_class in ['positif', 'netral', 'negatif']:
                # Get texts for this sentiment class
                class_texts = [text for text, label in zip(preprocessed_texts, labels) if label == sentiment_class]
                
                if class_texts:
                    # Create a document-term matrix
                    class_vectorizer = CountVectorizer(max_features=20)
                    dtm = class_vectorizer.fit_transform(class_texts)
                    
                    # Get feature names and their frequencies
                    feature_names = class_vectorizer.get_feature_names_out()
                    frequencies = np.sum(dtm.toarray(), axis=0)
                    
                    # Create a DataFrame for visualization
                    word_freq = [(word, freq) for word, freq in zip(feature_names, frequencies)]
                    word_freq.sort(key=lambda x: x[1], reverse=True)
                    top_words = word_freq[:15]  # Top 15 words
                    
                    if top_words:  # Only create visualization if we have words
                        # Create bar chart
                        plt.figure(figsize=(10, 6))
                        words, freqs = zip(*top_words)
                        plt.barh(words, freqs)
                        plt.xlabel('Frequency')
                        plt.ylabel('Words')
                        plt.title(f'Top Words in {sentiment_class.capitalize()} Reviews')
                        plt.tight_layout()
                        
                        # Save the visualization
                        word_freq_path = os.path.join('static', 'images', f'word_freq_{sentiment_class}.png')
                        plt.savefig(word_freq_path)
                        plt.close()
        except Exception as e:
            app.logger.error(f"Word frequency visualization error: {str(e)}")
    
    # 5. Calculate summary statistics for each folder
    summary = {}
    train_summary = {}
    test_summary = {}
    
    # Calculate grouped summary statistics
    for folder_name in grouped_results:
        grouped_summary[folder_name] = {}
        
        for group_key, items in grouped_results[folder_name].items():
            group_total = len(items)
            positif_count = sum(1 for item in items if item['sentiment'] == 'positif')
            netral_count = sum(1 for item in items if item['sentiment'] == 'netral')
            negatif_count = sum(1 for item in items if item['sentiment'] == 'negatif')
            
            grouped_summary[folder_name][group_key] = {
                'total': group_total,
                'positif': positif_count,
                'netral': netral_count,
                'negatif': negatif_count,
                'positif_percent': round((positif_count / group_total) * 100) if group_total > 0 else 0,
                'netral_percent': round((netral_count / group_total) * 100) if group_total > 0 else 0,
                'negatif_percent': round((negatif_count / group_total) * 100) if group_total > 0 else 0
            }
    
    for folder_name in sentiment_results:
        # All data summary
        all_items = sentiment_results[folder_name]
        folder_total = len(all_items)
        positif_count = sum(1 for item in all_items if item['sentiment'] == 'positif')
        netral_count = sum(1 for item in all_items if item['sentiment'] == 'netral')
        negatif_count = sum(1 for item in all_items if item['sentiment'] == 'negatif')
        
        summary[folder_name] = {
            'total': folder_total,
            'positif': positif_count,
            'netral': netral_count,
            'negatif': negatif_count,
            'positif_percent': round((positif_count / folder_total) * 100) if folder_total > 0 else 0,
            'netral_percent': round((netral_count / folder_total) * 100) if folder_total > 0 else 0,
            'negatif_percent': round((negatif_count / folder_total) * 100) if folder_total > 0 else 0
        }
        
        # Training data summary
        train_items = train_results[folder_name]
        train_total = len(train_items)
        train_positif = sum(1 for item in train_items if item['sentiment'] == 'positif')
        train_netral = sum(1 for item in train_items if item['sentiment'] == 'netral')
        train_negatif = sum(1 for item in train_items if item['sentiment'] == 'negatif')
        
        train_summary[folder_name] = {
            'total': train_total,
            'positif': train_positif,
            'netral': train_netral,
            'negatif': train_negatif,
            'positif_percent': round((train_positif / train_total) * 100) if train_total > 0 else 0,
            'netral_percent': round((train_netral / train_total) * 100) if train_total > 0 else 0,
            'negatif_percent': round((train_negatif / train_total) * 100) if train_total > 0 else 0
        }
        
        # Testing data summary
        test_items = test_results[folder_name]
        test_total = len(test_items)
        test_positif = sum(1 for item in test_items if item['sentiment'] == 'positif')
        test_netral = sum(1 for item in test_items if item['sentiment'] == 'netral')
        test_negatif = sum(1 for item in test_items if item['sentiment'] == 'negatif')
        
        test_summary[folder_name] = {
            'total': test_total,
            'positif': test_positif,
            'netral': test_netral,
            'negatif': test_negatif,
            'positif_percent': round((test_positif / test_total) * 100) if test_total > 0 else 0,
            'netral_percent': round((test_netral / test_total) * 100) if test_total > 0 else 0,
            'negatif_percent': round((test_negatif / test_total) * 100) if test_total > 0 else 0
        }
        
        # Generate sentiment distribution visualization for this folder
        try:
            plt.figure(figsize=(10, 6))
            labels = ['Positif', 'Netral', 'Negatif']
            sizes = [summary[folder_name]['positif_percent'], 
                    summary[folder_name]['netral_percent'], 
                    summary[folder_name]['negatif_percent']]
            colors = ['#22c55e', '#94a3b8', '#ef4444']
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(f'Sentiment Distribution for {folder_name}')
            
            # Save the visualization
            pie_chart_path = os.path.join('static', 'images', f'sentiment_dist_{folder_name.lower().replace(" ", "_")}.png')
            plt.savefig(pie_chart_path)
            plt.close()
        except Exception as e:
            app.logger.error(f"Pie chart visualization error: {str(e)}")
    
    # 6. Calculate overall summary
    total_items = sum(summary[folder]['total'] for folder in summary)
    total_positif = sum(summary[folder]['positif'] for folder in summary)
    total_netral = sum(summary[folder]['netral'] for folder in summary)
    total_negatif = sum(summary[folder]['negatif'] for folder in summary)
    
    overall_summary = {
        'total': total_items,
        'positif': total_positif,
        'netral': total_netral,
        'negatif': total_negatif,
        'positif_percent': round((total_positif / total_items) * 100) if total_items > 0 else 0,
        'netral_percent': round((total_netral / total_items) * 100) if total_items > 0 else 0,
        'negatif_percent': round((total_negatif / total_items) * 100) if total_items > 0 else 0
    }
    
    # Generate overall sentiment distribution visualization
    try:
        plt.figure(figsize=(10, 6))
        labels = ['Positif', 'Netral', 'Negatif']
        sizes = [overall_summary['positif_percent'], 
                overall_summary['netral_percent'], 
                overall_summary['negatif_percent']]
        colors = ['#22c55e', '#94a3b8', '#ef4444']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Overall Sentiment Distribution')
        
        # Save the visualization
        overall_pie_chart_path = os.path.join('static', 'images', 'overall_sentiment_dist.png')
        plt.savefig(overall_pie_chart_path)
        plt.close()
    except Exception as e:
        app.logger.error(f"Overall pie chart visualization error: {str(e)}")
    
    # 7. Calculate confusion matrix and metrics based on test data
    cm, metrics = calculate_metrics(all_true_labels, all_predictions)
    
    # 8. Save the sentiment data for future reference
    sentiment_data_id = str(uuid.uuid4())
    sentiment_file_path = os.path.join(app.config['TEMP_FOLDER'], f"sentiment_{sentiment_data_id}.json")
    
    # Convert numpy types before serialization to avoid JSON error
    data_to_save = convert_numpy_types({
        'data': sentiment_results,
        'train_data': train_results,
        'test_data': test_results,
        'summary': summary,
        'train_summary': train_summary,
        'test_summary': test_summary,
        'overall_summary': overall_summary,
        'grouped_results': grouped_results,
        'grouped_summary': grouped_summary,
        'confusion_matrix': cm,
        'metrics': metrics
    })
    
    with open(sentiment_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f)
    
    session['sentiment_data_id'] = sentiment_data_id
    
    # 9. Render the sentiment analysis template
    return render_template('sentiment_analysis.html', 
                          sentiment_results=sentiment_results,
                          train_results=train_results,
                          test_results=test_results,
                          summary=summary,
                          train_summary=train_summary,
                          test_summary=test_summary,
                          overall_summary=overall_summary,
                          grouped_results=grouped_results,
                          grouped_summary=grouped_summary,
                          confusion_matrix=cm,
                          metrics=metrics,
                          visualization_paths={
                              'confusion_matrix': '/static/images/confusion_matrix.png',
                              'metrics_comparison': '/static/images/metrics_comparison.png',
                              'word_freq_positif': '/static/images/word_freq_positif.png',
                              'word_freq_netral': '/static/images/word_freq_netral.png',
                              'word_freq_negatif': '/static/images/word_freq_negatif.png',
                              'overall_sentiment': '/static/images/overall_sentiment_dist.png'
                          })

# Register the route
@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment_route():
    return analyze_sentiment(app)

if __name__ == '__main__':
   app.config['DEBUG'] = True
   app.run(debug=True)