# models.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import os
from utils import mask_pii


def preprocess_data(df):
    """Apply PII masking and clean data."""
    df["email"], _ = zip(*df["email"].apply(mask_pii))
    return df


# Train the model
def train_model(data_path, save_dir="models/"):
    """Train a classification model on support emails."""
    # 1. Load the dataset
    df = pd.read_csv("combined_emails_with_natural_pii.csv")

    # 2. Preprocessing
    print("Preprocessing data...")
    df = preprocess_data(df)

    # 3. Feature and target
    X = df["email"]
    y = df["type"]

    # 4. Encode the labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save label encoder
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 6. Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 7. Model Initialization
    model = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        random_state=42, class_weight="balanced"
    )

    # 8. Train the model
    print("Training model...")
    model.fit(X_train_vec, y_train)

    # 9. Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 10. Save model and vectorizer
    print(f"Saving model and vectorizer to {save_dir}...")
    joblib.dump(model, os.path.join(save_dir, "classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(save_dir, "vectorizer.pkl"))

    print("Training Completed Successfully!")


# Function to load the # Function to save model, vectorizer, and label encoder
def save_model(model, vectorizer, label_encoder, save_dir="models/"):
    """Save the model, vectorizer, and label encoder to disk."""
    print(f"Saving model and vectorizer to {save_dir}...")
    joblib.dump(model, os.path.join(save_dir, "classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(save_dir, "vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.pkl"))

    print("Training Completed Successfully!")


# Function to load the trained model, vectorizer, and label encoder
def load_model(save_dir="models/"):
    """Load trained model, vectorizer, and label encoder."""
    model = joblib.load(os.path.join(save_dir, "classifier.pkl"))
    vectorizer = joblib.load(os.path.join(save_dir, "vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(save_dir, "label_encoder.pkl"))
    return model, vectorizer, label_encoder


# Function to classify email based on the trained model
def classify_email(
    email_body: str, model=None, vectorizer=None, label_encoder=None
) -> str:
    """Classify an email into a category using the trained model."""

    if model is None or vectorizer is None or label_encoder is None:
        # Load the model, vectorizer, and label encoder if not passed
        model, vectorizer, label_encoder = load_model()

    # Step 1: Preprocess the email body (vectorization)
    email_tfidf = vectorizer.transform([email_body])

    # Step 2: Predict the category using the trained model
    category_index = model.predict(email_tfidf)[0]

    # Step 3: Decode the predicted category using the label encoder (if needed)
    category_name = label_encoder.inverse_transform([category_index])[0]

    return category_name


# Assuming you have a trained model saved as 'email_classifier.pkl'
# You can now call classify_email with the necessary input
email_body = "Your email content here."
model, vectorizer, label_encoder = load_model()

# Classify email
category = classify_email(email_body, model, vectorizer, label_encoder)
print(f"Predicted email category: {category}")
