import argparse
import copy
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyrootutils
import torch
from medclip import MedCLIPVisionModel
from PIL import Image
from rich import print

# from sentence_transformers import SentenceTransformer
# from sentence_transformers import SentenceTransformer
from torchvision.ops import roi_align
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

ann = pd.read_feather("C:/Users/DryLab/Desktop/ViLLA/data/open-i/annotations.feather")

# Split ann into train and val
train_ann = ann[ann["split"] == "train"]
val_ann = ann[ann["split"] == "val"]

# Define attributes
ATTRIBUTES = [
    "cardiomegaly",
    "atelectasis",
    "opacities",
    "disease",
    "opacity",
    "limits",
    "normal limits",
    "granuloma",
    "degenerative changes",
    "effusions",
    "effusion",
    "emphysema",
    "changes",
    "markings",
    "granulomas",
    "calcifications",
    "midline",
    "density",
    "tortuosity",
    "sternotomy",
    "hiatal hernia",
    "lymph",
    "prominence",
    "deformity",
    "calcification",
    "osteophytes",
    "elevation",
    "edema",
    "surgical clips",
    "congestion",
    "nodules",
    "pneumonia",
    "enlargement",
    "clips",
    "hyperexpansion",
    "scoliosis",
    "copd",
    "dextroscoliosis",
    "calcified granuloma",
    "eventration",
    "pneumothorax",
    "tip",
    "consolidation",
    "aeration",
    "lucency",
    "ectasia",
    "deformities",
    "mild degenerative changes",
    "loss",
    "emphysematous changes",
]


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def generate_attribute_embs(out_dir):
    """
    Generate embeddings for each attribute.

    Parameters:
        out_dir: Directory for storing attribute embeddings
    """

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.cuda()

    final_sentences = []
    for attribute in ATTRIBUTES:
        # Split the attribute into individual words
        attribute_words = attribute.lower().split()

        # Identify sentences that contain any word from the attribute
        sentences_with_attributes = []

        for i in range(len(train_ann["sentences"])):
            for j in range(len(train_ann["sentences"].iloc[i])):
                sentence = train_ann["sentences"].iloc[i][j].lower()
                # Check if any word from attribute_words is in the sentence
                if any(word in sentence for word in attribute_words):
                    sentences_with_attributes.append(train_ann["sentences"].iloc[i][j])

        # Count the frequency of each sentence containing the attribute
        sentence_counter = Counter(sentences_with_attributes)

        # Get the 200 most frequent sentences
        most_common_sentences = [
            sentence for sentence, _ in sentence_counter.most_common(200)
        ]

        # Append the result for each attribute
        final_sentences.append(most_common_sentences)

    attribute_embeddings = []
    with torch.no_grad():
        for sentence in final_sentences:
            # Compute sentence embeddings
            sentences = tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to("cuda")

            sentence_embeddings = model.forward(**sentences)

            # sentence_embeddings = model.encode(sentence, convert_to_tensor=True)

            # Mean pooling
            average_embedding = mean_pooling(
                sentence_embeddings,
                sentences["attention_mask"],
            )

            # Average the embeddings
            average_embedding = sentence_embeddings.pooler_output.mean(
                axis=0, keepdims=True
            )
            average_embedding /= average_embedding.norm(dim=-1, keepdim=True)
            print(f"Average Embeddings Shape: {average_embedding.shape}")
            attribute_embeddings.append(average_embedding)
        else:
            print(f"No sentences found for attribute: {attribute}")

        attribute_embeddings = torch.stack(attribute_embeddings).squeeze()

    # Project embeddings to 1024 shape
    projection = torch.nn.Linear(average_embedding.shape[-1], 1024).to("cuda")

    attribute_embeddings = projection(attribute_embeddings)
    attribute_embeddings = attribute_embeddings.detach().cpu().numpy()

    attr_to_emb = dict(zip(ATTRIBUTES, attribute_embeddings))

    torch.save(attr_to_emb, f"{out_dir}/attr_embs.pth")
    print(f"Saved {len(attr_to_emb)} attribute embeddings to {out_dir}/attr_embs.pth")


def process_image(filepath):
    image = Image.open(filepath).convert("L")

    # Convert Image object to numpy array
    image = np.array(image)

    # Normalize image
    image = image.astype(np.float32) / 255.0

    # Apply histogram equalization
    image = cv2.equalizeHist((image * 255).astype(np.uint8)) / 255.0

    # Resize image
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(torch.float32)

    return image_tensor


def generate_region_embs(out_dir):
    """
    Generate embeddings for each region.

    Parameters:
        out_dir: Directory for storing region embeddings
    """
    ann = pd.read_feather(f"{out_dir}/annotations.feather")
    components = {}
    out_dir = Path(out_dir) / "region_embs"
    if out_dir.exists() and out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    # Load MedCLIP vision encoder
    # processor = MedCLIPProcessor()
    model = MedCLIPVisionModel()
    model.load_from_medclip("C:/Users/DryLab/Desktop/ViLLA/pretrained/medclip-resnet")
    image_encoder = model
    image_encoder = image_encoder.to("cuda")

    for layer in ["layer3", "layer4", "avgpool", "fc"]:
        components[layer] = copy.deepcopy(eval(f"image_encoder.model.{layer}"))
        setattr(image_encoder.model, layer, torch.nn.Identity())
    # for layer in ["layer4", "fc"]:
    #     setattr(image_encoder.model, layer, torch.nn.Identity())

    # Generate embeddings for each region
    reg_emb_map = {"image_id": [], "file": [], "file_id": []}
    all_reg_embs = []
    for idx, row in tqdm(ann.iterrows()):
        image_id, filepath = row["image_id"], row["image_filepath"]
        image = process_image(filepath).cuda()
        # image = Image.open(filepath).convert("L")
        # original_width, original_height = image.size
        # image = processor(images=image, return_tensors="pt")["pixel_values"][0].cuda()
        image = torch.stack([image])
        # regions = [
        #     resize_bounding_box(bbox, original_width, original_height)
        #     for bbox in row["region_bbox"]
        # ]
        regions = np.stack(row["region_bbox"].tolist())

        with torch.no_grad():
            features = image_encoder(image).reshape(1, 512, 28, 28)
            rois = (
                torch.cat((torch.zeros((len(regions), 1)), torch.tensor(regions)), 1)
                .to(torch.float32)
                .cuda()
            )
            x = roi_align(
                features,
                rois.to(dtype=features.dtype),
                (14, 14),
                features.shape[-1] / image.shape[-1],
                0,
                True,
            )

            x = components["layer3"](x)
            # x = components["layer4"](x)
            x = components["avgpool"](x).flatten(1)
            # x = components["fc"](x)
            reg_embs = x

        if len(all_reg_embs) == 2000:
            all_reg_embs = np.array(all_reg_embs, dtype=object)
            print(f"Shape of all embeddings: {all_reg_embs.shape}")
            np.savez_compressed(out_dir / f"embs_{idx}", all_reg_embs)
            reg_emb_map["file"].extend([f"embs_{idx}"] * all_reg_embs.shape[0])
            reg_emb_map["file_id"].extend(np.arange(all_reg_embs.shape[0]))
            all_reg_embs = []
        reg_embs = reg_embs.cpu().numpy()
        all_reg_embs.append(
            reg_embs.reshape(
                -1,
            )
        )
        reg_emb_map["image_id"].append(image_id)

    all_reg_embs = np.array(all_reg_embs, dtype=object)
    np.savez_compressed(out_dir / f"embs_{idx}", all_reg_embs)
    reg_emb_map["file"].extend([f"embs_{idx}"] * all_reg_embs.shape[0])
    reg_emb_map["file_id"].extend(np.arange(all_reg_embs.shape[0]))

    pd.DataFrame(reg_emb_map).to_feather(out_dir / "region_emb_mapping.feather")
    print(f"Saved region embeddings to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing functions for DocMNIST."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="Directory where DocMNIST data is stored (e.g. docmnist_30000_15.2)",
    )
    args = parser.parse_args()

    print("Generating attribute embeddings")
    generate_attribute_embs(os.path.join(root, "data", args.data_dir))
    print("-----------")

    print("Generating region embeddings")
    generate_region_embs(os.path.join(root, "data", args.data_dir))
    print("-----------")


if __name__ == "__main__":
    main()
