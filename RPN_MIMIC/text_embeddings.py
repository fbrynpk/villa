import os
from collections import Counter

import pandas as pd
import spacy
import torch
from radgraph import RadGraph
from sklearn.model_selection import train_test_split

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Check if a GPU is available and require it for spaCy
spacy.require_gpu()

# Load Spacy's English tokenizer
nlp = spacy.load("en_core_web_sm")

# Initialize RadGraph
radgraph = RadGraph()


def process_directory(main_dir):
    data = []

    for subdir, _, files in os.walk(main_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    data.append(
                        {
                            "text_path": os.path.relpath(file_path, start=main_dir),
                            "subfolder": os.path.splitext(file)[0],
                            "text_content": text,
                        }
                    )

    df = pd.DataFrame(data)
    return df


text_data_path = (
    "C:/Users/DryLab/Desktop/ViLLA/RPN_MIMIC/mimic-cxr-reports-preprocessed-villa/p10"
)

df = process_directory(text_data_path)
df = df[df["text_content"].apply(lambda x: isinstance(x, str) and len(x.split()) >= 2)]

def extract_entities(textual_descriptions):
    extracted_entities = []

    textual_descriptions = radgraph(textual_descriptions)

    for _, description in textual_descriptions.items():
        # Extract entities using RadGraph
        entities = description.get("entities", {})

        # Filter entities that are 'Definitely Present'
        for _, entities_label in entities.items():
            label = entities_label.get("label")
            tokens = entities_label.get("tokens")
            if label == "Observation::definitely present":
                extracted_entities.append(tokens)

    return extracted_entities


def filter_nouns(entities):
    noun_entities = []

    for entity in entities:
        # Tokenize the entity using Spacy
        doc = nlp(entity)
        for token in doc:
            # Filter out tokens that are nouns
            if token.pos_ == "NOUN":
                # Append the noun to the list of noun entities
                noun_entities.append(token.text)

    return noun_entities


df["tokens"] = df["text_content"].apply(extract_entities)
df["noun_tokens"] = df["tokens"].apply(filter_nouns)

# Split data into training and validation and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.6667, random_state=42, shuffle=True)

all_noun_tokens = [token for sublist in train_df["noun_tokens"] for token in sublist]
noun_token_counts = Counter(all_noun_tokens)
token = noun_token_counts.most_common(50)

attributes = []

for i in range(0, 50):
    attributes.append(token[i][0])


# Save attributes to a text file
output_file_path = "C:/Users/DryLab/Desktop/ViLLA/RPN_MIMIC/top_50_attributes_final.txt"
with open(output_file_path, "w", encoding="utf-8") as file:
    for attribute in attributes:
        file.write(f"{attribute}\n")

print(f"Attributes saved to {output_file_path}")
