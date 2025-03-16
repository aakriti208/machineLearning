import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Retain only necessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to binary (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# print(df.head())

# Download stopwords if not already downloaded
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenize words
    words = text.split()
    # Remove stopwords and stem words
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply preprocessing
df["processed_message"] = df["message"].apply(preprocess_text)


X_train, X_test, y_train, y_test = train_test_split(
    df["processed_message"], df["label"], test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(X_train_tfidf.shape, X_test_tfidf.shape)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Model training completed.")

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print(df.head())

def predict_spam(message):
    processed = preprocess_text(message)
    transformed = vectorizer.transform([processed])
    prediction = model.predict(transformed)
    return "Spam" if prediction[0] == 1 else "Ham"

print(predict_spam("Congratulations! You've won a free gift. Click now!"))
print(predict_spam("Hey,whadup?"))

print(df.head())