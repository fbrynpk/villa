import torch
from rich import print

from villa.dataloaders.Stage1Dataset import Stage1Dataset

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
# [
#     "limits",
#     "changes",
#     "atelectasis",
#     "disease",
#     "opacities",
#     "opacity",
#     "calcifications",
#     "granulomas",
#     "tip",
#     "midline",
#     "clips",
#     "effusion",
#     "sternotomy",
#     "density",
#     "tortuosity",
#     "size",
#     "markings",
#     "lymph",
#     "appearance",
#     "calcification",
#     "effusions",
#     "deformity",
#     "elevation",
#     "prominence",
#     "borderline",
#     "xxxx",
#     "edema",
#     "osteophytes",
#     "eventration",
#     "nodules",
#     "scoliosis",
#     "congestion",
#     "hyperinflation",
#     "pneumonia",
#     "curvature",
#     "enlargement",
#     "consolidation",
#     "densities",
#     "lucency",
#     "copd",
#     "hyperexpansion",
#     "fusion",
#     "engorgement",
#     "pneumothorax",
#     "deformities",
#     "aeration",
#     "process",
#     "cabg",
#     "age",
#     "indeterminate",
# ]

# [
#     "pneumothorax",
#     "effusion",
#     "limits",
#     "disease",
#     "changes",
#     "consolidation",
#     "effusions",
#     "opacities",
#     "findings",
#     "atelectasis",
#     "opacity",
#     "abnormality",
#     "edema",
#     "abnormalities",
#     "nodules",
#     "calcifications",
#     "tip",
#     "appearance",
#     "density",
#     "sternotomy",
#     "size",
#     "clips",
#     "pneumothoraces",
#     "process",
#     "air",
#     "masses",
#     "markings",
#     "granulomas",
#     "deformity",
#     "pneumonia",
#     "calcification",
#     "elevation",
#     "tortuosity",
#     "lymph",
#     "midline",
#     "prominence",
#     "osteophytes",
#     "borderline",
#     "eventration",
#     "congestion",
#     "lucency",
#     "cabg",
#     "scoliosis",
#     "copd",
#     "hyperinflation",
#     "enlargement",
#     "areas",
#     "scar",
#     "curvature",
#     "arthritic",
# ]


class OPEN_Stage1Dataset(Stage1Dataset):
    def __init__(self, split: str, data_dir: str):
        """
        Initialize OPEN dataloader for Stage 1.

        Parameters:
            split (str): Indicates the split (e.g. "train", "val")
            data_dir (str): Directory where OPEN data is stored
        """

        super(OPEN_Stage1Dataset, self).__init__(split, data_dir)

        self.examples.insert(
            self.examples.shape[1],
            "attr_binary_vec",
            self.encode_attributes(ATTRIBUTES),
        )

        self.region_embs = self.get_region_embs()

        print(f"=> Split {self.split} includes {self.__len__()} samples")

    def __len__(self):
        return self.examples.shape[0]

    def __getitem__(self, idx: int):
        example = self.examples.iloc[[idx]]
        image_id = str(example["image_id"].values[0])

        out_dict = {
            "idx": torch.tensor(idx),
            "image_id": image_id,
        }

        out_dict.update(self.get_inputs(example, image_id))
        return out_dict


if __name__ == "__main__":
    pass
