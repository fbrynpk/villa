import re
from collections import Counter

import pandas as pd
import spacy
import torch
from radgraph import RadGraph
from rich import print
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Spacy's English tokenizer
nlp = spacy.load("en_core_web_sm")

# Initialize RadGraph
radgraph = RadGraph()

DATA_DIR = "C:/Users/DryLab/Desktop/villa/data/open-i"

df = pd.read_feather(
    "C:/Users/DryLab/Desktop/ViLLA/data/open-i/annotations_region_final.feather"
)

text_path_list = df["text_filepath"].tolist()


def filter_sentences_by_absent_tokens(radgraph_output):
    remaining_sentences = []

    radgraph_output = radgraph(radgraph_output)

    # Iterate over RadGraph output
    for _, description in radgraph_output.items():
        text = description.get("text", " ")
        entities = description.get("entities", {})

        # Step 1: Split the text into sentences
        doc = nlp(text)
        sentences = [sent.text.lower() for sent in doc.sents]

        # Step 2: Create a mapping of word indices to sentence indices
        word_to_sentence = {}
        current_word_idx = 0

        for i, sentence in enumerate(sentences):
            sentence_words = sentence.split()
            for _ in sentence_words:
                word_to_sentence[current_word_idx] = i
                current_word_idx += 1

        # Step 3: Create a flag to keep or discard each sentence
        sentence_flags = [True] * len(sentences)

        # Step 4: Iterate over entities and check for tokens labeled "Observation::definitely absent"
        for _, entity_info in entities.items():
            label = entity_info.get("label")
            start_ix = entity_info.get("start_ix")

            # Check if the entity is labeled as "Observation::definitely absent"
            if label == "Observation::definitely absent":
                # Step 5: Find which sentence contains the token based on start and end indices
                sentence_idx = word_to_sentence.get(start_ix, None)
                if sentence_idx is not None:
                    sentence_flags[sentence_idx] = False

        # Step 6: Collect the remaining sentences
        for i, sentence in enumerate(sentences):
            if sentence_flags[i]:
                if not re.match(r"^\d\s*\.$", sentence.strip()):
                    remaining_sentences.append(sentence)

        # Revert back to the original text
        remaining_sentences = " ".join(remaining_sentences)

    return remaining_sentences


def clean_text(text):
    return re.sub(r"\b[xX]+\b", "[MASK]", text)


def extract_entities(textual_descriptions):
    extracted_entities = []

    # RadGraph output processing
    textual_descriptions = radgraph(
        textual_descriptions
    )  # Assuming radgraph() is your entity extractor

    for _, description in textual_descriptions.items():
        entities = description.get("entities", {})

        # Step 1: Identify all real entities (Observation::definitely present with no direct modify relations)
        real_entities = {
            entity_id: entity_data
            for entity_id, entity_data in entities.items()
            if entity_data.get("label") == "Observation::definitely present"
        }

        filter_entities = {
            entity_id: entity_data
            for entity_id, entity_data in entities.items()
            if entity_data.get("label") == "Observation::definitely present"
            and not any(
                "modify" in relation for relation in entity_data.get("relations", [])
            )
        }

        # Step 2: Create a map of modifiers to the entities they modify
        modifier_map = {}
        for entity_id, entity_data in entities.items():
            relations = entity_data.get("relations", [])
            for relation in relations:
                if relation[0] == "modify" and relation[1] in entities:
                    modifier_map.setdefault(relation[1], []).append(entity_id)

        # Step 3: Recursive function to gather all modifiers for each real entity
        def get_all_modifiers(entity_id):
            visited = set()  # To track visited entities
            stack = [entity_id]  # Initialize stack with the starting entity
            modifiers = []

            while stack:
                current_entity_id = stack.pop()

                if current_entity_id in visited:
                    continue  # Skip already visited entities

                visited.add(current_entity_id)  # Mark current entity as visited

                # Add current entity's modifiers to the list
                direct_modifiers = modifier_map.get(current_entity_id, [])
                for modifier_id in direct_modifiers:
                    modifier_data = entities[modifier_id]
                    modifiers.append(
                        (modifier_data.get("tokens"), modifier_data.get("start_ix"))
                    )
                    stack.append(
                        modifier_id
                    )  # Add the modifier to the stack for further exploration

            return modifiers

        # Step 4: Combine real entities with all their modifiers
        for entity_id, entity_data in real_entities.items():
            observation_text = entity_data.get("tokens")
            # print(observation_text)
            observation_start_ix = entity_data.get("start_ix")
            # Gather all modifiers recursively
            modifiers_for_this_entity = get_all_modifiers(entity_id)

            # Create a list of tokens, including both entity and modifier tokens
            all_tokens = [
                (observation_text, observation_start_ix)
            ] + modifiers_for_this_entity

            # Sort all tokens by their start_ix to ensure correct order
            sorted_tokens = sorted(all_tokens, key=lambda x: x[1])

            # Extract the sorted tokens (ignoring the start_ix for final output)
            sorted_entity_text = " ".join([token[0] for token in sorted_tokens]).strip()

            # Ensure only real entities with their full modifier chains are added to the output
            if (
                observation_text in filter_entities.get(entity_id)["tokens"]
                if entity_id in filter_entities
                else False
            ):
                extracted_entities.append(sorted_entity_text)

    # Return unique extracted entities (no duplicates)
    return list(set(extracted_entities))


def filter_nouns(entities):
    noun_entities = []

    for entity in entities:
        # Tokenize the entity using Spacy
        doc = nlp(entity)

        # Extract noun phrases rather than individual nouns
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]

        # Append the noun phrases to the list
        noun_entities.extend(noun_phrases)

    return noun_entities


def split_into_sentences(text):
    # Tokenize the text into sentences
    doc = nlp(text)

    # Extract sentences
    sentences = [sent.text for sent in doc.sents]

    return sentences


df["text"] = df["text_filepath"].apply(lambda x: open(x, "r").read())

# Break down the text into sentences and filter out the sentences that contain tokens labeled as "Observation::definitely absent"
df["text"] = df["text"].apply(filter_sentences_by_absent_tokens)

df["text"] = df["text"].apply(clean_text)

df["attributes"] = df["text"].apply(extract_entities)

df["attributes"] = df["attributes"].apply(filter_nouns)

# # Remove row if no attributes are present
mask = df["attributes"].apply(lambda x: len(x) > 0)

df["attributes"] = df["attributes"].where(mask)

df = df.dropna(subset=["attributes"])

# Split the data into training and test sets (70% training and 30% test)
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

# Split the remaining data into validation and test sets (10% validation and 20% test)
# val_df, test_df = train_test_split(
#     split_df, test_size=2 / 3, random_state=42, shuffle=True
# )

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

train_df["split"] = "train"
val_df["split"] = "val"

attributes_token = [item for sublist in train_df["attributes"] for item in sublist]

count_attributes = Counter(attributes_token)

most_common_attributes = count_attributes.most_common(50)

attributes = []

for i in range(len(most_common_attributes)):
    attributes.append(most_common_attributes[i][0])

# Save attributes to txt
with open("attributes.txt", "w") as f:
    for item in attributes:
        f.write("%s\n" % item)

print(attributes)

train_df["sentences"] = train_df["text"].apply(split_into_sentences)
val_df["sentences"] = val_df["text"].apply(split_into_sentences)

frames = [train_df, val_df]

df = pd.concat(frames)

df["attributes"] = df["attributes"].apply(
    lambda x: [attr for attr in x if attr in attributes]
)

# Remove row if no attributes are less than 3
mask = df["attributes"].apply(lambda x: len(x) > 0)

df["attributes"] = df["attributes"].where(mask)

df = df.dropna(subset=["attributes"])

df = df[
    [
        "image_id",
        "image_size",
        "image_filepath",
        "region_coord",
        "region_bbox",
        "region_labels",
        "num_regions",
        "text_filepath",
        "text",
        "sentences",
        "attributes",
        "split",
    ]
]

id_mapping = {
    id_value: idx for idx, id_value in enumerate(sorted(df["image_id"]), start=0)
}
# Map the original image_id to the new ordered index
df["image_id"] = df["image_id"].map(id_mapping)
df["image_id"] = range(len(df))
df.reset_index(inplace=True)
df.drop(columns=["index"], inplace=True)

df.to_feather(f"{DATA_DIR}/annotations.feather")
