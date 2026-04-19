"""
SpamShield — Flask Backend
Serves the /predict API and history endpoints.
Auto-trains the model on first launch if no saved model exists.
"""

import os
import re
import sys
import json
import pickle
import string
import sqlite3
import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'spam_model.pkl')
DB_PATH    = os.path.join(BASE_DIR, 'database', 'spamshield.db')

# ---------------------------------------------------------------------------
# Ensure model exists
# ---------------------------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("🔧  No trained model found — training now …")
        sys.path.insert(0, os.path.join(BASE_DIR, 'model'))
        from train_model import train_and_save
        return train_and_save()
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

# ---------------------------------------------------------------------------
# NLTK / Preprocessing  (mirrors train_model.py)
# ---------------------------------------------------------------------------
# Built-in English stopwords (no NLTK required)
STOPWORDS = frozenset([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
    'won','wouldn'
])

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email_text TEXT    NOT NULL,
            prediction TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            keywords   TEXT    NOT NULL,
            created_at TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_result(email_text, prediction, confidence, keywords):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO history (email_text, prediction, confidence, keywords, created_at) VALUES (?,?,?,?,?)",
        (
            email_text[:2000],          # cap stored text
            prediction,
            round(confidence, 4),
            json.dumps(keywords),
            datetime.datetime.utcnow().isoformat()
        )
    )
    conn.commit()
    conn.close()

def fetch_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            'id':         r['id'],
            'email_text': r['email_text'],
            'prediction': r['prediction'],
            'confidence': r['confidence'],
            'keywords':   json.loads(r['keywords']),
            'created_at': r['created_at'],
        })
    return result

def fetch_stats():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    spam  = conn.execute("SELECT COUNT(*) FROM history WHERE prediction='spam'").fetchone()[0]
    ham   = conn.execute("SELECT COUNT(*) FROM history WHERE prediction='ham'").fetchone()[0]
    conn.close()
    return {'total': total, 'spam': spam, 'ham': ham}

# ---------------------------------------------------------------------------
# Explainability — top TF-IDF spam-contributing words
# ---------------------------------------------------------------------------
def get_spam_keywords(pipeline, raw_text: str, top_n: int = 8):
    """
    Returns words from the input that most strongly pushed the
    prediction toward 'spam' according to Logistic Regression coefficients.
    """
    try:
        vectorizer = pipeline.named_steps['tfidf']
        clf        = pipeline.named_steps['clf']

        # Class index for 'spam'
        classes = list(clf.classes_)
        spam_idx = classes.index('spam') if 'spam' in classes else 1

        # Coefficients for spam class
        coef = clf.coef_[0] if len(clf.coef_) == 1 else clf.coef_[spam_idx]

        # Transform the input
        processed = preprocess_text(raw_text)
        tfidf_vec  = vectorizer.transform([processed])
        feature_names = vectorizer.get_feature_names_out()

        # Non-zero features in this document
        cx = tfidf_vec.tocoo()
        scored = {}
        for _, col, val in zip(cx.row, cx.col, cx.data):
            feature = feature_names[col]
            # Only single-word tokens for highlighting
            if ' ' not in feature:
                scored[feature] = float(coef[col]) * val

        # Sort by contribution to spam
        keywords = sorted(scored, key=lambda w: scored[w], reverse=True)[:top_n]
        return keywords
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app    = Flask(__name__)
CORS(app)

# Load model once at startup
print("🚀  Loading SpamShield model …")
MODEL = load_model()
init_db()
print("✅  SpamShield is ready!")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: { "text": "email content here" }
    Returns: { "prediction", "confidence", "keywords", "risk_level" }
    """
    data = request.get_json(silent=True) or {}
    email_text = (data.get('text') or '').strip()

    if not email_text:
        return jsonify({'error': 'No text provided'}), 400

    processed = preprocess_text(email_text)

    # Predict + probability
    prediction   = MODEL.predict([processed])[0]          # 'spam' or 'ham'
    proba        = MODEL.predict_proba([processed])[0]    # [ham_prob, spam_prob]

    classes      = list(MODEL.classes_)
    spam_idx     = classes.index('spam') if 'spam' in classes else 1
    spam_conf    = float(proba[spam_idx])
    confidence   = spam_conf if prediction == 'spam' else 1 - spam_conf

    # Risk level
    if spam_conf < 0.40:
        risk_level = 'low'
    elif spam_conf < 0.70:
        risk_level = 'medium'
    else:
        risk_level = 'high'

    # Explainability keywords
    keywords = get_spam_keywords(MODEL, email_text)

    # Persist
    save_result(email_text, prediction, confidence, keywords)

    return jsonify({
        'prediction':  prediction,
        'confidence':  round(confidence * 100, 1),
        'spam_score':  round(spam_conf * 100, 1),
        'risk_level':  risk_level,
        'keywords':    keywords,
    })


@app.route('/history', methods=['GET'])
def history():
    """GET /history — last 50 analyzed emails."""
    limit = min(int(request.args.get('limit', 50)), 200)
    return jsonify(fetch_history(limit))


@app.route('/stats', methods=['GET'])
def stats():
    """GET /stats — aggregate counts."""
    return jsonify(fetch_stats())


@app.route('/history/<int:record_id>', methods=['DELETE'])
def delete_history(record_id):
    """DELETE /history/:id — remove a single record."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return jsonify({'deleted': True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
