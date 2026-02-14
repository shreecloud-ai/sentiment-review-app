import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Paths (relative to project root)
DATA_PATH = "notebooks/data/cleaned_reviews.csv"
MODEL_PATH = "models/sentiment_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"  # optional, but we'll save separately

print("Loading cleaned data...")
df = pd.read_csv(DATA_PATH)

# We use 'full_review' as input, 'sentiment' as target
X = df['full_review']
y = df['sentiment']

# Train-test split (80/20) – stratified to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print("Class distribution in train:\n", y_train.value_counts(normalize=True).round(3))

# Build pipeline: TF-IDF → Logistic Regression
# Build pipeline: TF-IDF → Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=3,
        max_df=0.95
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        random_state=42,
        # multi_class='multinomial',   ← REMOVE or comment this line
        n_jobs=-1,
        class_weight='balanced'      # ← highly recommended in your case (see below)
    ))
])

print("\nTraining model...")
pipeline.fit(X_train, y_train)

# Evaluate on test set
y_pred = pipeline.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=3))

# Save the entire pipeline (easiest way – contains both vectorizer + model)
joblib.dump(pipeline, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")