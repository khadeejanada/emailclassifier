from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from utils import mask_pii, restore_pii  # Import restore_pii
from models import classify_email

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Hello World"}
# Pydantic model to define the structure of the request body


class EmailRequest(BaseModel):
    email_body: str
# Pydantic model to define the structure of masked entities


class MaskedEntity(BaseModel):
    position: List[int]  # [start_index, end_index]
    classification: str  # e.g., "full_name", "email", "phone_number"
    entity: str  # the original entity (value before masking)
# Pydantic model to define the structure of the response


class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str
    restored_email: str  # Add restored_email to the response


@app.post("/classify_email/", response_model=EmailResponse)
async def classify(request: EmailRequest):
    email_body = request.email_body
    # Step 1: Mask the PII in the email
    masked_email, masked_entities = mask_pii(email_body)
    # Step 2: Classify the email (you should replace this with your own model)
    category = classify_email(
        masked_email
    )  # Assuming classify_email takes the masked email as input

    # Step 3: Restore the original PII in the email
    restored_email = restore_pii(masked_email, masked_entities)

    # Step 4: Prepare the response as per the required format
    response = EmailResponse(
        input_email_body=email_body,
        list_of_masked_entities=masked_entities,
        masked_email=masked_email,
        category_of_the_email=category,
        restored_email=restored_email,
    )

    return response
