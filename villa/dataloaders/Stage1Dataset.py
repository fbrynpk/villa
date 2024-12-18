import abc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Stage1Dataset(Dataset):
    __metaclass__ = abc.ABC

    def __init__(self, split, data_dir):
        """
        Initialize dataloader for Stage 1.

        Parameters:
            split (str): Indicates the split (e.g. "train", "val")
            data_dir (str): Directory where data is stored
        """
        super().__init__()

        self.split = split
        self.data_dir = Path(data_dir)

        self.ann = pd.read_feather(self.data_dir / "annotations.feather")
        self.examples = self.ann[self.ann["split"] == split]

    def encode_attributes(self, attributes):
        """
        Generate binary vector indicating the presence/absence of each attribute.

        Parameters:
            attributes (list): List of all attributes in the dataset
        Returns:
            attr_binary_vec (list): List of binary attribute presence/absence vectors
                                    for each image in the dataset
        """
        attr_binary_vec = []
        for idx, row in self.examples.iterrows():
            vec = np.zeros(len(attributes))
            for a in row["attributes"]:
                vec[attributes.index(a)] = 1
            attr_binary_vec.append(vec)
        return attr_binary_vec

    def get_region_embs(self):
        """
        Load precomputed region embeddings

        Returns:
            region_embs (dict): Maps each image id to corresponding region embeddings
        """
        region_embs = {}
        print(f"Loading region embeddings from {self.data_dir}/region_embs")
        emb_df = pd.read_feather(
            f"{self.data_dir}/region_embs/region_emb_mapping.feather"
        )
        curr_open_npz = None
        valid_samples = set(self.examples["image_id"].values.tolist())
        for idx, row in tqdm(emb_df.iterrows()):
            image_id = row["image_id"]
            if image_id not in valid_samples:
                continue
            if row["file"] != curr_open_npz:
                curr_open_npz = row["file"]
                embs = np.load(
                    f"{self.data_dir}/region_embs/{curr_open_npz}.npz",
                    allow_pickle=True,
                )["arr_0"]
            region_embs[str(image_id)] = embs[row["file_id"]].reshape(-1, 1024)
        return region_embs

    def get_inputs(self, example, image_id):
        """
        Helper function for __getitem__()

        Parameters:
            example (pd.DataFrame): Selected row from annotation dataframe
            image_id (str): Image identifier
        Returns:
            out_dict (dict): Dictionary populated with relevant inputs for stage 1 models
        """
        out_dict = {}
        reg_emb = self.region_embs[image_id]
        reg_emb = np.array(reg_emb, dtype=np.float32)
        out_dict["num_regions"] = torch.tensor(reg_emb.shape[0])
        out_dict["img"] = torch.tensor(reg_emb)
        out_dict["attr_labels"] = torch.tensor(
            example["attr_binary_vec"].values[0]
        ).unsqueeze(0)
        return out_dict
