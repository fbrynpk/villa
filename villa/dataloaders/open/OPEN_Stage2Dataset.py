import numpy as np
import torch
from PIL import Image
from rich import print

from villa.dataloaders.Stage2Dataset import Stage2Dataset


class OPEN_Stage2Dataset(Stage2Dataset):
    def __init__(self, split, data_dir, stage_1_ckpt_dir):
        """
        Initialize OPEN dataloader for Stage 1.

        Parameters:
            split (str): Indicates the split (e.g. "train", "val")
            data_dir (str): Directory where OPEN data is stored
            stage_1_ckpt_dir (str): Directory where checkpoints from stage 1 are stored
        """
        super().__init__(split, data_dir, stage_1_ckpt_dir)

        # Use results from Stage 1 to create a training dataset with image-text pairs
        # as well as learned region-attribute pairs
        self.create_train_dataset()

        # Precompute text embeddings
        text_emb, valid_region_idx = self.get_text_embs()
        self.examples = self.examples.assign(text_emb=text_emb)
        self.examples = self.examples.assign(valid_region_idx=valid_region_idx)

        print(f"Final {self.split} dataset includes {self.examples.shape[0]} samples")

    def __len__(self):
        return self.examples.shape[0]

    def __getitem__(self, idx):
        example = self.examples.iloc[[idx]]
        image_id = str(example["image_id"].values[0])
        image_path = example["image_filepath"].values[0]

        out_dict = {
            "idx": torch.tensor(idx),
            "image_id": image_id,
        }

        # Load image and associated regions
        image = Image.open(image_path).convert("RGB")

        # Resize image
        image = image.resize((224, 224))

        # Convert Image object to numpy array
        image = np.array(image)

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image).to(torch.float32)

        # Rearrange dimensions to (B, C, H, W)
        image = image.permute(2, 0, 1)

        out_dict["image"] = torch.stack([image])
        out_dict["num_regions"] = torch.tensor(
            len(example["valid_region_idx"].values[0])
        )

        regions = torch.tensor(np.stack(example["region_coord"].values[0]))
        # print(example["valid_region_idx"].values[0])
        out_dict["region"] = regions[example["valid_region_idx"].values[0], :]

        # Load text embedding
        txt_emb = torch.tensor(example["text_emb"].values[0])
        out_dict["txt_emb"] = txt_emb / txt_emb.norm(dim=1, keepdim=True)

        return out_dict

    def create_train_dataset(self):
        """
        Augment the training dataset to include image-text pairs in addition to region-attribute
        pairs
        """
        assigned_text = self.examples.apply(
            lambda x: x.assigned_text.tolist() + [x.text], axis=1
        )
        self.examples = self.examples.assign(assigned_text=assigned_text)
        self.examples = self.examples.assign(num_regions=lambda x: x.num_regions + 1)
        region_coord = self.examples.apply(
            lambda x: x.region_bbox.tolist()
            + [np.array([0, 0, x.image_size[1] - 1, x.image_size[0] - 1])],
            axis=1,
        )
        self.examples = self.examples.assign(region_coord=region_coord)


if __name__ == "__main__":
    pass
