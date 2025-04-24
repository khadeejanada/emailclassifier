from api import app  # This is your FastAPI app
from models import load_model
from utils import mask_pii


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
# Load the model, vectorizer, and label encoder
model, vectorizer, label_encoder = load_model("models/")


def classify_email(email_text):
    # Step 1: Mask PII and get only the masked text
    masked_text, _ = mask_pii(email_text)

    # Step 2: Vectorize the text
    email_vector = vectorizer.transform([masked_text])

    # Step 3: Predict and decode the label
    predicted_label = model.predict(email_vector)
    category = label_encoder.inverse_transform(predicted_label)[0]

    return category


if __name__ == "__main__":
    sample_email = """Subject: Data Analytics for Investment
    I am contacting you to request information on data analytics
    tools that can be utilized with the Eclipse IDE for enhancing
    investment optimization."""

    predicted_category = classify_email(sample_email)
    print(
        f"The predicted category for the email is: {predicted_category}")
