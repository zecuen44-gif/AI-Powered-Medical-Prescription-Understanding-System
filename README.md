# Intelligent Medical Prescription Understanding and Validation System using DONUT Transformer OCR

## Overview

This project is an AI-powered system for understanding handwritten medical prescriptions. It uses a pre-trained DONUT (Document Understanding Transformer) OCR model to extract text from prescription images and applies medicine validation using RapidFuzz-based fuzzy matching.

The system converts handwritten prescriptions into structured, machine-readable information by identifying medicines, extracting dosages, detecting prescription frequencies, and validating medicine names against a medicine database.

---

## Problem Statement

Doctors' handwritten prescriptions are often difficult to interpret due to:

- Illegible handwriting
- Abbreviated medicine names
- OCR spelling errors
- Manual interpretation errors

This project aims to automate prescription understanding and improve medicine recognition using AI and fuzzy matching techniques.

---

## Objectives

- Extract handwritten prescription text using DONUT OCR
- Identify medicine names from extracted text
- Correct OCR spelling errors using RapidFuzz
- Extract dosage information
- Detect medicine frequency information
- Generate structured prescription analysis

---

## System Architecture

Prescription Image
↓
DONUT Transformer OCR
↓
Text Extraction
↓
RapidFuzz Medicine Validation
↓
Medicine Database Matching
↓
Dosage Extraction
↓
Frequency Detection
↓
Structured Prescription Analysis

---

## Features

### OCR-Based Prescription Understanding

- Handwritten prescription recognition
- Printed prescription recognition
- Transformer-based document understanding

### Medicine Validation

- RapidFuzz fuzzy matching
- OCR error correction
- Medicine database lookup

Examples:

Azituro → Azithro

Certifizine → Cetirizine

Montelukast → Montelukast

### Dosage Extraction

Detects:

- 500mg
- 650mg
- 10mg
- 5ml
- 250mg

### Frequency Detection

Recognizes:

- OD (Once Daily)
- BD (Twice Daily)
- TDS (Three Times Daily)
- HS (At Bedtime)
- SOS (When Required)

### Structured Analysis

Generates:

- Extracted text
- Detected medicines
- Dosages
- Frequencies
- Classification confidence

---

## Technologies Used

### Artificial Intelligence

- DONUT Transformer OCR
- Hugging Face Transformers
- PyTorch

### Natural Language Processing

- RapidFuzz
- Zero-Shot Classification (BART Large MNLI)

### Backend

- Python

### Interface

- Gradio

### Data Processing

- Pandas
- Regular Expressions

---

## Medicine Database

The system uses a medicine knowledge base containing:

- Antibiotics
- Anti-allergy medicines
- Pain relievers
- Blood pressure medicines
- Diabetes medicines
- Acidity medicines
- Vitamin supplements

Examples:

- Paracetamol
- Cetirizine
- Levocetirizine
- Azithromycin
- Amoxicillin
- Cefixime
- Pantoprazole
- Metformin
- Telmisartan
- Amlodipine

---

## Project Workflow

1. User uploads prescription image.
2. DONUT OCR extracts prescription text.
3. Text is classified as medical prescription or non-prescription.
4. Medicine names are identified using RapidFuzz.
5. Dosages are extracted using pattern matching.
6. Frequencies are detected.
7. Structured analysis is displayed.

---

## Sample Output

Extracted Text:

Paracetamol 500mg BD

Cetirizine 10mg OD

Medicine Analysis:

Medicine:

- Paracetamol
- Cetirizine

Dosage:

- 500mg
- 10mg

Frequency:

- Twice Daily
- Once Daily

Classification:

- Medical Prescription

Confidence:

- 0.95

---

## Current Limitations

- OCR performance depends on handwriting quality.
- Very poor handwriting may reduce recognition accuracy.
- Medicine database currently contains common medicines only.
- Dosage and frequency extraction are rule-based.

---

## Future Enhancements

- Larger medicine database
- Streamlit dashboard
- FastAPI backend
- Medicine recommendation module
- Drug interaction checking
- Cloud deployment
- Prescription segmentation
- Fine-tuning on additional prescription datasets

---

## Applications

- Digital prescription management
- Pharmacy assistance
- Medical record digitization
- Healthcare document processing
- Prescription validation systems

---

## Conclusion

This project demonstrates how Transformer-based OCR and fuzzy matching can be combined to transform handwritten medical prescriptions into structured and understandable information. The system improves medicine recognition reliability through OCR validation and provides a foundation for future healthcare automation applications.
