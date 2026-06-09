import os
import re
import torch
import pandas as pd

from PIL import Image
from rapidfuzz import process

from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
    pipeline
)

import gradio as gr

# ======================================================
# PATHS
# ======================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
donut_model_path = os.path.join(current_dir, "model")

# ======================================================
# LOAD MEDICINE DATABASE
# ======================================================

db = pd.read_csv("medicine_database.csv")
medicine_names = db["name"].tolist()

# ======================================================
# LOAD DONUT MODEL
# ======================================================

try:

    processor = DonutProcessor.from_pretrained(
        donut_model_path
    )

    donut_model = VisionEncoderDecoderModel.from_pretrained(
        donut_model_path
    )

except Exception as e:

    print(f"Error loading Donut model: {e}")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

donut_model.to(device)
donut_model.eval()

# ======================================================
# LOAD BART CLASSIFIER
# ======================================================

try:

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if device == "cuda" else -1
    )

except Exception as e:

    print(f"Error loading classifier: {e}")
    exit(1)

candidate_labels = [
    "medical prescription",
    "not medical prescription"
]

# ======================================================
# OCR
# ======================================================

def extract_text_from_image(image):

    image = image.convert("RGB")

    encoding = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        generated_ids = donut_model.generate(
            encoding.pixel_values,
            max_length=512,
            num_beams=1,
            early_stopping=True,
            decoder_start_token_id=
            processor.tokenizer.convert_tokens_to_ids(
                "<s_ocr>"
            )
        )

    text = processor.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()

# ======================================================
# MEDICINE DETECTION
# ======================================================

IGNORE_WORDS = {
    "amit",
    "sharma",
    "doctor",
    "date",
    "review",
    "days",
    "after",
    "before",
    "daily",
    "long",
    "take",
    "twice",
    "once",
    "medicine",
    "mbbs",
    "md",
    "retrieved",
    "patient",
    "clinic",
    "hospital",
    "signature",
    "dr",
    "rx"
}

def find_medicines(text):

    words = text.split()

    detected = []

    for word in words:

        original_word = word

        word = word.strip(".,:-()[]").lower()

        if len(word) < 5:
            continue

        if word in IGNORE_WORDS:
            continue

        match = process.extractOne(
            word,
            medicine_names,
            score_cutoff=80
        )

        if match:

            medicine_name = match[0]
            score = match[1]

            detected.append(
                f"{original_word} → {medicine_name} ({score:.1f}%)"
            )

    return sorted(set(detected))

# ======================================================
# DOSAGE EXTRACTION
# ======================================================

def extract_dosages(text):

    dosages = re.findall(
        r'\d+\s?(?:mg|ml|g)',
        text,
        flags=re.IGNORECASE
    )

    return dosages

# ======================================================
# FREQUENCY EXTRACTION
# ======================================================

def extract_frequency(text):

    frequency_map = {

        "OD": "Once Daily",
        "BD": "Twice Daily",
        "TDS": "Three Times Daily",
        "HS": "At Bedtime",
        "SOS": "When Required"
    }

    results = []

    text_upper = text.upper()

    for key, value in frequency_map.items():

        if key in text_upper:

            results.append(value)

    return results

# ======================================================
# CLASSIFICATION
# ======================================================

medical_keywords = [

    "prescribed",
    "take",
    "mg",
    "ml",
    "capsules",
    "dosage",
    "dr.",
    "doctor",
    "patient",
    "medications",
    "apply",
    "signature",
    "clinic",
    "pharmacy",
    "rx",
    "dose",
    "medicine",
    "drug"
]

def classify_prescription_zero_shot(text):

    if not text:

        return "No text found", 0.0

    result = classifier(
        text,
        candidate_labels
    )

    predicted_label = result["labels"][0]
    confidence = result["scores"][0]

    text_lower = text.lower()

    has_medical_keywords = any(
        keyword in text_lower
        for keyword in medical_keywords
    )

    if predicted_label == "not medical prescription" and has_medical_keywords:

        predicted_label = "medical prescription"
        confidence = max(confidence, 0.75)

    elif predicted_label == "medical prescription" and not has_medical_keywords:

        predicted_label = "not medical prescription"
        confidence = max(confidence, 0.75)

    return predicted_label, confidence

# ======================================================
# MAIN PIPELINE
# ======================================================

def classify_image_gradio(image):

    extracted_text = extract_text_from_image(
        image
    )

    predicted_label, confidence = classify_prescription_zero_shot(
        extracted_text
    )

    medicines = find_medicines(
        extracted_text
    )

    dosages = extract_dosages(
        extracted_text
    )

    frequency = extract_frequency(
        extracted_text
    )

    medicine_report = f"""

Detected Medicines
------------------

{chr(10).join(medicines) if medicines else "None Found"}

Dosages
-------

{chr(10).join(dosages) if dosages else "None Found"}

Frequency
---------

{chr(10).join(frequency) if frequency else "None Found"}

"""

    return (
        extracted_text,
        medicine_report,
        predicted_label,
        round(confidence, 3)
    )

# ======================================================
# UI
# ======================================================

demo = gr.Interface(

    fn=classify_image_gradio,

    inputs=gr.Image(type="pil"),

    outputs=[

        gr.Textbox(
            label="Extracted Text",
            lines=8
        ),

        gr.Textbox(
            label="Medicine Analysis",
            lines=12
        ),

        gr.Textbox(
            label="Predicted Label"
        ),

        gr.Number(
            label="Confidence Score"
        )
    ],

    title="Intelligent Medical Prescription Understanding System",

    description="""
Upload a prescription image.
The system performs:

1. DONUT OCR
2. Prescription Classification
3. Medicine Detection
4. Dosage Extraction
5. Frequency Detection
"""
)

# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )