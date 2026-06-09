from rapidfuzz import process
import pandas as pd
import re

db = pd.read_csv("medicine_database.csv")

medicine_names = db["name"].tolist()

def find_medicines(text):

    words = text.split()

    found = []

    for word in words:

        match = process.extractOne(
            word,
            medicine_names
        )

        if match and match[1] > 80:
            found.append(match[0])

    return list(set(found))


def extract_dosages(text):

    return re.findall(
        r'\d+\s?(mg|ml|g)',
        text,
        flags=re.IGNORECASE
    )


def extract_frequency(text):

    freq_map = {
        "OD":"Once Daily",
        "BD":"Twice Daily",
        "TDS":"Three Times Daily",
        "HS":"At Bedtime",
        "SOS":"When Required"
    }

    results = []

    text_upper = text.upper()

    for key,value in freq_map.items():

        if key in text_upper:
            results.append(value)

    return results