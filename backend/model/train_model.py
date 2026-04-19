"""
SpamShield - Model Training Script
Trains a Logistic Regression classifier with TF-IDF features
on a built-in spam/ham dataset.
"""

import os
import re
import pickle
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Built-in dataset (augmented for variety & coverage)
# ---------------------------------------------------------------------------
SPAM_SAMPLES = [
    "Congratulations! You've won a $1,000,000 prize! Click here to claim now!",
    "FREE MONEY! You are our lucky winner today. Call us immediately to collect.",
    "URGENT: Your bank account has been suspended. Verify now at this link.",
    "Buy cheap Viagra online! No prescription needed. Best prices guaranteed!",
    "Make money fast from home! Earn $5000 per week working just 2 hours a day!",
    "You have been selected for a special offer. Act now before it expires!",
    "Weight loss miracle pill! Lose 30 pounds in 30 days guaranteed!",
    "Investment opportunity! 500% returns guaranteed. Send Bitcoin now.",
    "Your PayPal account is limited. Click to restore access immediately.",
    "You won the international lottery! Claim your $500,000 prize today!",
    "Get out of debt fast! We can eliminate your debt completely!",
    "CLICK HERE to get a FREE iPhone 15 Pro. Limited time offer!",
    "Hot singles in your area want to meet you tonight! Click here!",
    "Exclusive deal for you ONLY! Buy now and save 90% off everything!",
    "Your computer has a virus! Download our free antivirus immediately.",
    "Double your income in 30 days! Our proven system guarantees results.",
    "Congratulations, you have been pre-approved for a $50,000 loan!",
    "Send this email to 10 friends and win a vacation package worth $10,000!",
    "Limited offer: FREE casino chips worth $500. Sign up now!",
    "Increase your credit score instantly! Our secret method works fast!",
    "Work from home opportunity! No experience needed. Earn thousands daily!",
    "Your inheritance of $4.5 million awaits. Contact us for transfer details.",
    "WINNER ANNOUNCEMENT: Your email has won £1,500,000. Claim prize now.",
    "Cheap meds delivered overnight. No prescription. 100% discreet.",
    "ACT NOW: Last chance to claim your free gift card worth $200!",
    "We noticed unusual activity on your account. Verify your details NOW.",
    "Get rich quick with our proven crypto trading bot. 300% profits daily!",
    "You owe IRS back taxes. Call immediately or face arrest warrant.",
    "Special bonus: deposit $100 get $1000 free. Casino guaranteed winnings.",
    "Your email address has been chosen for a $750 Walmart gift card!",
    "Lose weight without diet or exercise! This pill melts fat overnight.",
    "Earn passive income while you sleep! Our automated system does all work.",
    "FINAL WARNING: Your account will be deleted unless you verify now!",
    "Cheap loans approved instantly! Bad credit OK. No questions asked.",
    "Buy followers and likes! 10,000 Instagram followers for just $5!",
    "Free trial of our amazing product! Just pay $1 shipping today.",
    "You are the 1000th visitor! You won a MacBook Pro. Claim it now!",
    "Enlarge your earnings! Our forex signals make 1000 pips per week.",
    "CONFIDENTIAL: I need your help transferring $10 million out of Nigeria.",
    "Meet beautiful ladies online! Free registration for men. Join now!",
    "Your subscription expires today. Renew now to avoid service disruption.",
    "Send $500 in gift cards to unlock your $25,000 sweepstakes prize!",
    "Miracle cure for diabetes! Doctors HATE this one weird trick.",
    "Warning: Your device has been hacked! Download protection software NOW!",
    "Exclusive VIP membership! Earn commissions recommending our products.",
]

HAM_SAMPLES = [
    "Hi John, are we still on for the meeting tomorrow at 3 PM?",
    "Please find attached the quarterly report for your review.",
    "I wanted to follow up on our conversation from last week.",
    "The project deadline has been moved to next Friday. Please update your schedules.",
    "Can you send me the updated budget spreadsheet when you get a chance?",
    "Happy birthday! Hope you have a wonderful day.",
    "The package you ordered has been shipped and will arrive in 3-5 days.",
    "Thank you for your application. We will be in touch soon.",
    "Just a reminder about the team lunch tomorrow at noon.",
    "I've reviewed the document and have a few comments. Let's discuss tomorrow.",
    "Could you please confirm your attendance at the conference?",
    "The IT department will be performing maintenance this weekend.",
    "Attached are the meeting notes from today's discussion.",
    "I'm running 10 minutes late. Please start without me.",
    "The new software update is now available for download.",
    "We'd like to invite you to our annual company picnic next Saturday.",
    "Your appointment has been confirmed for Tuesday at 2:30 PM.",
    "Please review the attached contract before our meeting.",
    "The library books you borrowed are due back next week.",
    "Great work on the presentation! The client was very impressed.",
    "I'll be out of office from Monday to Wednesday. Contact Sarah for urgent matters.",
    "The server will be down for scheduled maintenance from 2-4 AM Sunday.",
    "Your flight has been confirmed. Check-in opens 24 hours before departure.",
    "We're looking forward to seeing you at the conference next week.",
    "The monthly newsletter is attached. Please share with your team.",
    "Can we reschedule our call to 4 PM instead? I have a conflict at 3.",
    "Your order #12345 has been delivered to your front door.",
    "Thanks for the feedback. I'll incorporate your suggestions into the next draft.",
    "The homework assignment is due by midnight Friday.",
    "Congratulations on your promotion! Well deserved.",
    "I found the missing report in the shared drive under Q3 folder.",
    "Please remember to bring your ID to the office tomorrow for badge renewal.",
    "The weather forecast shows rain this weekend. Plan accordingly.",
    "Your password will expire in 7 days. Please reset it at your convenience.",
    "Team standup is moved to 10 AM tomorrow due to the all-hands meeting.",
    "I'll send you the invoice once the project is complete.",
    "The book club meets every Thursday at 7 PM at the community center.",
    "Your car is ready for pickup at the service center.",
    "We've updated our privacy policy. Please review the changes.",
    "Reminder: Annual performance reviews begin next month.",
    "The charity fundraiser exceeded our goal! Thank you to all who contributed.",
    "Please submit your expense reports by end of day Friday.",
    "I enjoyed our lunch meeting today. Looking forward to our collaboration.",
    "The new employee orientation will be held in conference room B.",
    "Your prescription is ready for pickup at the pharmacy.",
]

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# Built-in English stopwords (no NLTK download required)
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
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove punctuation
      3. Tokenize
      4. Remove stopwords
    """
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Remove extra whitespace / digits-only tokens
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # 4. Tokenize & remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_and_save():
    print("🔧  Preparing dataset …")
    texts  = SPAM_SAMPLES + HAM_SAMPLES
    labels = ['spam'] * len(SPAM_SAMPLES) + ['ham'] * len(HAM_SAMPLES)

    processed = [preprocess_text(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("🤖  Training Logistic Regression + TF-IDF pipeline …")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n✅  Accuracy: {acc*100:.1f}%")
    print(classification_report(y_test, y_pred))

    # Persist
    model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, 'spam_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"💾  Model saved → {model_path}")
    return pipeline


if __name__ == '__main__':
    train_and_save()
