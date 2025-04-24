import re
from typing import List, Tuple, Dict


# Function to mask PII data and keep track of the original values and positions
def mask_pii(email_body: str) -> Tuple[str, List[Dict]]:
    """
    This function accepts an email body, masks PII information, and returns:
    - The email body with PII masked.
    - A list of dictionaries containing details about the masked PII entities.
    """
    # Define regex patterns for different PII types (just examples)
    patterns = {
        "full_name": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z|a-z]{2,}\b",
        "phone_number": r"\b\d{10}\b",  # Match 10-digit phone numbers
        "dob": r"\b\d{2}/\d{2}/\d{4}\b",  # Match date of birth (MM/DD/YYYY)
        "aadhar_num": r"\b\d{12}\b",  # Match Aadhaar numbers (12 digits)
        "credit_debit_no": r"\b\d{16}\b",  # Match credit/debit card numbers
        "cvv_no": r"\b\d{3}\b",  # Match CVV codes (3 digits)
        "expiry_no": r"\b\d{2}/\d{2}\b",  # Match expiry dates (MM/YY)
    }

    # Initialize the masked email and entities list
    masked_email = email_body
    entities = []
    offset = 0  # Keeps track of position shift caused by replacements

    # Iterate over each PII type and apply the masking logic
    for entity_type, pattern in patterns.items():
        for match in re.finditer(pattern, email_body):
            original = match.group(0)
            start = match.start() + offset
            end = match.end() + offset

            # Create a masked token to replace the PII
            masked_token = f"[{entity_type}]"
            masked_email = masked_email[:start] + masked_token
            + masked_email[end:]

            # Store the details of the masked entity for future restoration
            entities.append(
                {
                    "position": [start, start + len(masked_token)],
                    "classification": entity_type,
                    "entity": original,
                }
            )

            # Update the offset to account for the length change
            offset += len(masked_token) - len(original)

    return masked_email, entities


# Function
def restore_pii(masked_email: str, masked_entities: List[Dict]) -> str:
    """
    This function restores the original PII entities back to the masked email.
    It ensures that the email body is restored by replacing each masked token
    with its original PII.
    """

    # Sort entities
    masked_entities.sort(key=lambda x: x["position"][0], reverse=True)

    for entity in masked_entities:
        start, end = entity["position"]
        original_entity = entity["entity"]
        # Replace the masked entity with the original one
        masked_email = (
            masked_email[:start] +
            original_entity +
            masked_email[end:]
        )

    return masked_email

# Optional: Function to debugging


def log_masking_details(
    email_body: str,
    masked_email: str,
    entities: List[Dict]
):
    """
    This function logs the details of the masking process.
    Useful for debugging and understanding the transformations applied.
    """
    print("Original Email Body:\n", email_body)
    print("\nMasked Email Body:\n", masked_email)
    print("\nEntities Masked:")
    for entity in entities:
        print(
            f"Entity: {entity['entity']}, "
            f"Type: {entity['classification']}, "
            f"Position: {entity['position']}"
        )
