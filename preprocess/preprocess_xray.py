import argparse
import copy
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyrootutils
import torch
from medclip import MedCLIPModel, MedCLIPProcessor, MedCLIPTextModel, MedCLIPVisionModel
from PIL import Image
from rich import print
from torch.nn import functional as F
from tqdm.auto import tqdm

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

ann = pd.read_feather("C:/Users/DryLab/Desktop/ViLLA/data/xray/annotations.feather")


def generate_attribute_embs(out_dir):
    """
    Generate embeddings for each attribute.

    Parameters:
        out_dir: Directory for storing attribute embeddings
    """
    attribute_embeddings = []

    processor = MedCLIPProcessor()
    model = MedCLIPTextModel()
    model.cuda()

    attributes = [
        "infiltration",
        "opacities",
        "borderline",
        "markings",
        "formation",
        "change",
        "place",
        "tube",
        "scoliosis",
        "infiltrations",
        "elevation",
        "effusion",
        "fixation",
        "gas",
        "calcification",
        "consolidation",
        "shadow",
        "bronchopneumonia",
        "kyphoscoliosis",
        "patches",
        "opacity",
        "plaques",
        "port",
        "tip",
        "arteriosclerosis",
        "densities",
        "tortuosity",
        "endotracheal",
        "patch",
        "ekg",
        "atherosclerosis",
        "faint",
        "catheter",
        "fracture",
        "fibrosis",
        "inspiration",
        "intubation",
        "disease",
        "sternotomy",
        "pattern",
        "glass",
        "ground",
        "osteophytes",
        "tortuousity",
        "p",
        "calcifications",
        "hyperinflation",
        "enlargement",
        "blunt",
        "congestion",
    ]

    for attribute in attributes:
        # Identify sentences that contain the attribute from the "sentences" column
        sentences_with_attributes = []

        for i in range(len(ann["sentences"])):
            for j in range(len(ann["sentences"].iloc[i])):
                sentence = ann["sentences"].iloc[i][j]
                if attribute in sentence:
                    sentences_with_attributes.append(sentence)

        # Count the frequency of each sentence containing the attribute
        sentence_counter = Counter(sentences_with_attributes)

        # Get the 200 most frequent sentences
        most_common_sentences = [
            sentence for sentence, _ in sentence_counter.most_common(200)
        ]

        with torch.inference_mode():
            if most_common_sentences:
                # Compute sentence embeddings
                most_common_sentences = processor(
                    text=most_common_sentences, return_tensors="pt", padding=True
                ).to("cuda")
                sentence_embeddings = model.forward(
                    most_common_sentences["input_ids"],
                    most_common_sentences["attention_mask"],
                )

                # Average the embeddings
                average_embedding = sentence_embeddings.mean(axis=0, keepdim=True)
                # average_embedding /= average_embedding.norm(dim=-1, keepdim=True)
                attribute_embeddings.append(average_embedding)
            else:
                print(f"No sentences found for attribute: {attribute}")

    attribute_embeddings = (
        torch.stack(attribute_embeddings).squeeze().detach().cpu().numpy()
    )

    # attribute_embeddings = np.stack(attribute_embeddings)
    # attribute_embeddings = torch.tensor(attribute_embeddings).to(device)
    # print(attribute_embeddings.shape)
    attr_to_emb = dict(zip(attributes, attribute_embeddings))

    torch.save(attr_to_emb, f"{out_dir}/attr_embs.pth")
    print(f"Saved {len(attr_to_emb)} attribute embeddings to {out_dir}/attr_embs.pth")


def process_image(filepath):
    image = Image.open(filepath).convert("L")

    # Resize image
    image = image.resize((224, 224))

    # Convert Image object to numpy array
    image = np.array(image)

    # Normalize image
    image = image.astype(np.float32) / 255.0

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(torch.float32)

    return image_tensor


def generate_region_embs(out_dir):
    """
    Generate embeddings for each region using MedCLIP.

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
    model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
    model.from_pretrained()
    image_encoder = model.vision_model
    image_encoder = image_encoder.to("cuda")

    for layer in ["layer3", "avgpool"]:
        components[layer] = copy.deepcopy(getattr(image_encoder.model, layer))
        setattr(image_encoder.model, layer, torch.nn.Identity())
    for layer in ["layer4", "fc"]:
        setattr(image_encoder.model, layer, torch.nn.Identity())

    # Prepare to store the embeddings
    reg_emb_map = {"image_id": [], "file": [], "file_id": []}
    all_reg_embs = []

    # Generate embeddings for each region
    for idx, row in tqdm(ann.iterrows(), total=len(ann)):
        image_id, filepath = row["image_id"], row["image_filepath"]
        # Load and resize the image
        image = process_image(filepath).cuda()
        regions = row["region_coord"]  # These are the multiclass masks

        with torch.no_grad():
            # Generate image features
            features = image_encoder(image)

            # Initialize a list to store region embeddings for each class
            all_class_embs = []

            for mask in regions:
                # Convert mask to a tensor
                mask_tensor = torch.tensor(
                    mask.reshape(224, 224), dtype=torch.float32
                ).cuda()

                # print(f"Mask tensor shape: {mask_tensor.shape}")

                # Resize the mask to match the feature map size
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=features.shape[-2:],
                    mode="nearest",
                )

                # print(f"Mask tensor after reshape: {mask_tensor.shape}")

                # Apply the mask to the feature map
                masked_features = features * mask_tensor
                # print(f"Masked features shape: {masked_features.shape}")
                # Perform global average pooling on the masked features
                masked_features = components["layer3"](masked_features)
                pooled_features = (
                    components["avgpool"](masked_features).squeeze(-1).squeeze(-1)
                )
                # print(f"Pooled features shape: {pooled_features.shape}")
                all_class_embs.append(pooled_features)

            # Concatenate embeddings for all classes
            reg_embs = torch.cat(all_class_embs, dim=0).cpu().numpy()
            # print(f"Final region embeddings shape: {reg_embs.shape}")
            all_reg_embs.append(reg_embs.reshape(-1))

            if len(all_reg_embs) >= 1000:
                all_reg_embs = np.array(all_reg_embs, dtype=object)
                print(f"Shape of all embeddings: {all_reg_embs.shape}")
                np.savez_compressed(
                    out_dir / f"embs_{idx}", all_reg_embs, allow_pickle=True
                )
                reg_emb_map["file"].extend([f"embs_{idx}"] * all_reg_embs.shape[0])
                reg_emb_map["file_id"].extend(np.arange(all_reg_embs.shape[0]))
                all_reg_embs = []

            reg_emb_map["image_id"].append(image_id)

    # Save any remaining embeddings
    if all_reg_embs:
        all_reg_embs = np.array(all_reg_embs, dtype=object)
        np.savez_compressed(out_dir / f"embs_{idx}", all_reg_embs, allow_pickle=True)
        reg_emb_map["file"].extend([f"embs_{idx}"] * all_reg_embs.shape[0])
        reg_emb_map["file_id"].extend(np.arange(all_reg_embs.shape[0]))

    # Save the mapping of region embeddings
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

    # print("Generating region embeddings")
    # generate_region_embs(os.path.join(root, "data", args.data_dir))
    # print("-----------")


if __name__ == "__main__":
    main()
