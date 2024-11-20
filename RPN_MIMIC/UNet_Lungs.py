import os

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from segmentation_models_pytorch import utils
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class LungsDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # Load image
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Load mask

        # Convert to NumPy array
        image = np.array(image)
        mask = np.array(mask)

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply histogram equalization
        image = cv2.equalizeHist((image * 255).astype(np.uint8)) / 255.0

        # Resize the image and mask to (224, 224)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

        # Optionally apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert image and mask to PyTorch tensors
        image = torch.from_numpy(image).unsqueeze(0).float()  # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0).long()

        return image, mask


# Define the transformation using albumentations
transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.5),
        A.Affine(),
    ]
)


# Path to lungs images and masks
image_dir = "C:/Users/DryLab/Desktop/ViLLA/RPN_MIMIC/data/Images/lungs_images"
mask_dir = "C:/Users/DryLab/Desktop/ViLLA/RPN_MIMIC/data/Images/lungs_mask"

# Get list of image and mask files
image_paths = sorted(
    [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
)
mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

# Split data into train, validation, and test sets into 0.8, 0.1, 0.1
train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = (
    train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
)
val_image_paths, test_image_paths, val_mask_paths, test_mask_paths = train_test_split(
    test_image_paths, test_mask_paths, test_size=0.5, random_state=42
)

# Create dataset
train_heart_dataset = LungsDataset(
    train_image_paths,
    train_mask_paths,
    transform=transform,
)

val_heart_dataset = LungsDataset(val_image_paths, val_mask_paths, transform=None)
test_heart_dataset = LungsDataset(test_image_paths, test_mask_paths, transform=None)

# Create dataloader
train_dataloader = DataLoader(
    train_heart_dataset, batch_size=16, shuffle=True, num_workers=4
)
val_dataloader = DataLoader(
    val_heart_dataset, batch_size=16, shuffle=False, num_workers=4
)
test_dataloader = DataLoader(
    test_heart_dataset, batch_size=16, shuffle=False, num_workers=4
)


def get_losses():
    return train_logs_list, valid_logs_list


if __name__ == "__main__":
    # Set device agnositc code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3,
        activation=None,
    )

    # Define loss function, optimizer, and
    criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    criterion.__name__ = "DiceLoss"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics = [utils.metrics.IoU(threshold=0.5)]

    TRAINING = True

    EPOCHS = 15

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_epoch = utils.train.TrainEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model, loss=criterion, metrics=metrics, device=device, verbose=True
    )

    if TRAINING:
        best_loss = 1.0
        train_logs_list, valid_logs_list = [], []

        print(f"Training model using {device}")

        for i in range(0, EPOCHS):
            # Perform training and validation
            print("\nEpoch: {}".format(i))
            train_logs = train_epoch.run(train_dataloader)
            valid_logs = valid_epoch.run(val_dataloader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            # Print logs to confirm keys and values
            print(f"Validation Logs: {valid_logs}")

            # Check for the correct IoU score key
            current_loss = valid_logs.get("DiceLoss", None)
            current_iou = valid_logs.get("iou_score", None)
            if current_loss is not None:
                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save(model, "./UNet_Lungs_Multiclass.pth")
                    print(
                        f"New best loss score: {best_loss} with an IoU score of {current_iou}. Model saved!"
                    )
            else:
                print("IoU score not found in logs.")
