import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich import print
from scipy.ndimage import label
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

DATA_DIR = "C:/Users/DryLab/Desktop/villa/data/open-i"


class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")

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


def extract_id(path):
    basename = os.path.basename(path)

    # Extract the ID from the filename
    return "_".join(basename.split("_")).replace(".txt", "").replace(".png", "")


def lungs_segmentation(model, dataloader, device):
    print(f"Segmenting lung images using {device}")
    model.eval()
    segmentation_results = []

    with torch.inference_mode():
        for batch, (images) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            model = model.to(device)

            masks = model(images)
            masks = masks.argmax(dim=1).cpu().numpy()

            for i in range(images.shape[0]):
                image_id = batch * dataloader.batch_size + i
                segmentation_results.append((image_id, masks[i]))

    return segmentation_results


def heart_segmentation(model, dataloader, device):
    print(f"Segmenting heart images using {device}")
    model.eval()
    segmentation_results = []

    with torch.inference_mode():
        for batch, (images) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            model = model.to(device)

            masks = model(images)
            masks = (masks > 0.5).cpu().numpy().squeeze()

            for i in range(images.shape[0]):
                image_id = batch * dataloader.batch_size + i
                segmentation_results.append((image_id, masks[i]))

    return segmentation_results


def postprocess_mask(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)

    # Step 1: Binary opening
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (11, 11)
    )  # Disk-shaped structuring element with radius 5
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Step 2: Keep the largest contiguous segment
    labeled_mask, num_features = label(opened_mask)
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0  # Exclude background
    largest_segment = np.argmax(sizes)
    largest_mask = np.where(labeled_mask == largest_segment, 1, 0).astype(np.uint8)

    # Step 3: Binary fill holes
    filled_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_CLOSE, kernel)

    # Step 4: Binary dilation
    dilated_mask = cv2.dilate(filled_mask, kernel)

    return dilated_mask


def get_bounding_box(region_coords):
    # Find the indices where the region_label exists
    indices = np.nonzero(region_coords)

    # If there are no coordinates for the region_label, return a placeholder
    if len(indices[0]) == 0:
        return np.array([None, None, None, None])  # Placeholder for empty regions

    # Get the min and max x and y coordinates
    min_y, min_x = np.min(indices, axis=1)
    max_y, max_x = np.max(indices, axis=1)

    return np.array([min_x, min_y, max_x, max_y])


def process_row(region_coords, original_shape):
    # Reshape the region_coords to the original shape
    right_lung_coords = region_coords[0]  # Right Lung
    left_lung_coords = region_coords[1]  # Left Lung
    heart_coords = region_coords[2]  # Heart

    right_lung_2d = right_lung_coords.reshape(original_shape)
    left_lung_2d = left_lung_coords.reshape(original_shape)
    heart_2d = heart_coords.reshape(original_shape)

    # Calculate the bounding box for each region
    left_lung_bbox = get_bounding_box(left_lung_2d)
    right_lung_bbox = get_bounding_box(right_lung_2d)
    heart_bbox = get_bounding_box(heart_2d)

    # Concatenate all bounding boxes into a single 1D array
    region_bbox = [
        np.array(right_lung_bbox.reshape(-1)),
        np.array(left_lung_bbox.reshape(-1)),
        np.array(heart_bbox.reshape(-1)),
    ]

    # Initialize a NumPy array to store the reg_coords for this image
    reg_bbox_array = np.empty(3, dtype=object)

    # Store each organ's coordinates in the reg_coords_array
    for j in range(3):
        reg_bbox_array[j] = region_bbox[j]

    return reg_bbox_array


def main():
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir = "C:/Users/DryLab/Desktop/ConVIRT/Dataset/open-i/data/front"
    image_paths = sorted(
        [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)],
        key=lambda x: int(os.path.basename(x).split("CXR")[-1].split("_")[0]),
    )
    text_dir = "C:/Users/DryLab/Desktop/ConVIRT/Dataset/open-i/data/ecgen-radiology-txt"

    text_paths = sorted(
        os.listdir(text_dir),
        key=lambda filename: int(filename.split("_")[0].replace("CXR", "")),
    )

    text_paths = [os.path.join(text_dir, path) for path in text_paths]

    # Create dictionaries to map image IDs to text paths and vice versa
    text_dict = {extract_id(path): path for path in text_paths}
    image_dict = {extract_id(path): path for path in image_paths}

    common_ids = set(image_dict.keys()) & set(text_dict.keys())

    paired_data = []
    for common_id in common_ids:
        paired_data.append(
            {
                "image_filepath": image_dict[common_id],
                "text_filepath": text_dict[common_id],
            }
        )

    paired_data.sort(key=lambda x: x["image_filepath"])

    df = pd.DataFrame(paired_data)
    df["image_id"] = range(len(df))

    image_paths_list = df["image_filepath"].tolist()

    train_dataset = CustomDataset(image_paths_list, transform=None)
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )

    heart_model = torch.load(
        "C:/Users/DryLab/Desktop/ViLLA/UNet_Heart.pth", map_location=device
    )

    lungs_model = torch.load(
        "C:/Users/DryLab/Desktop/ViLLA/UNet_Lungs_Multiclass.pth", map_location=device
    )

    lungs_mask = lungs_segmentation(lungs_model, train_dataloader, device)

    heart_mask = heart_segmentation(heart_model, train_dataloader, device)

    # Post-Processing

    # Assuming lungs_mask contains (batch, mask) pairs
    for idx, (batch, mask) in enumerate(tqdm(lungs_mask)):
        # Define left and right lungs
        right_lung = (mask == 1).astype(np.uint8)
        left_lung = (mask == 2).astype(np.uint8)

        # Apply the existing postprocess_mask function
        postprocessed_right_lung = postprocess_mask(right_lung)
        postprocessed_left_lung = postprocess_mask(left_lung)

        # Combine the postprocessed masks back into one output
        postprocessed_mask = np.zeros_like(mask)
        postprocessed_mask[postprocessed_right_lung == 1] = 1
        postprocessed_mask[postprocessed_left_lung == 1] = 2

        # Update the mask in place
        lungs_mask[idx] = (batch, postprocessed_mask)

    # Assuming lungs_mask contains (batch, mask) pairs
    for idx, (batch, mask) in enumerate(tqdm(heart_mask)):
        # Apply the existing postprocess_mask function
        postprocessed_mask = postprocess_mask(mask)

        # Update the mask in place
        heart_mask[idx] = (batch, postprocessed_mask)

    df["image_size"] = [list(mask.shape) for _, mask in lungs_mask]

    # Combined Lungs and Heart Region

    full_mask = []

    for idx, (batch, masks) in enumerate(tqdm(lungs_mask)):
        # Create separate masks for right lung, left lung, and heart
        right_lung = (masks == 1).astype(np.uint8)
        left_lung = (masks == 2).astype(np.uint8)
        heart = (heart_mask[idx][1] == 1).astype(np.uint8)

        # Initialize combined_mask with zeros
        combined_mask = np.zeros_like(masks, dtype=np.uint8)

        # Assign distinct values to each region
        combined_mask[right_lung == 1] = 1  # Right Lung
        combined_mask[left_lung == 1] = 2  # Left Lung
        combined_mask[heart == 1] = 3  # Heart

        combined_mask = combined_mask.astype(np.uint8)

        # Append the batch and the combined mask to full_mask
        full_mask.append((batch, combined_mask))

    # Initialize a NumPy array to store reg_coords for all images
    all_reg_coords = np.empty(len(full_mask), dtype=object)

    for i in range(len(full_mask)):
        batch, mask = full_mask[i]

        # Convert mask to torch tensor for processing
        mask_tensor = torch.from_numpy(mask)

        # Create boolean masks for each organ
        right_lung_mask = mask_tensor == 1
        left_lung_mask = mask_tensor == 2
        heart_mask = mask_tensor == 3

        # Combine the coordinates into an array of arrays
        reg_coords = [
            np.array(right_lung_mask.cpu().numpy().astype(np.uint64).reshape(-1)),
            np.array(left_lung_mask.cpu().numpy().astype(np.uint64).reshape(-1)),
            np.array(heart_mask.cpu().numpy().astype(np.uint64).reshape(-1)),
        ]

        # Initialize a NumPy array to store the reg_coords for this image
        reg_coords_array = np.empty(3, dtype=object)

        # Store each organ's coordinates in the reg_coords_array
        for j in range(3):
            reg_coords_array[j] = reg_coords[j]

        # Append to the main NumPy array
        all_reg_coords[i] = reg_coords_array

    # Add DataFrame Column with the coordinates of the masks
    df["region_coord"] = all_reg_coords

    df["num_regions"] = df["region_coord"].apply(lambda x: len(x))

    # Assuming `all_reg_coords` is already populated as in your previous example
    # and `original_shape` is known for each image

    # Initialize a NumPy array to store region_bbox for all images
    all_region_bbox = np.empty(len(all_reg_coords), dtype=object)

    for i in range(len(all_reg_coords)):
        region_coords = all_reg_coords[i]

        # Assuming original_shape is known and is the same for all images
        # Replace `original_shape` with the actual shape of the mask
        original_shape = (224, 224)  # Replace with actual dimensions

        # Process each row to compute bounding boxes
        region_bbox = process_row(region_coords, original_shape)

        # Store the result in the NumPy array
        all_region_bbox[i] = region_bbox

    # Apply function to each row
    df["region_bbox"] = df["region_coord"].apply(
        lambda x: process_row(x, original_shape)
    )

    df["region_labels"] = [np.array([1, 2, 3]) for _ in range(len(df))]

    df["region_labels"] = df["region_labels"].apply(lambda labels: [labels])

    # Initialize an empty NumPy array to store the transformed region labels
    transformed_labels = np.empty(len(df), dtype=object)

    # Process each row in the DataFrame
    for i in range(len(df)):
        # Extract the nested array
        nested_array = df["region_labels"].iloc[i][0]

        # Initialize an empty NumPy array to store the transformed labels
        labels_array = np.empty(len(nested_array), dtype=object)

        # Convert each label to its own NumPy array and store it
        for j, labels in enumerate(nested_array):
            labels_array[j] = np.array([labels], dtype=np.uint64)

        # Store the transformed labels array in the main array
        transformed_labels[i] = labels_array

    # Update the DataFrame with the transformed labels
    df["region_labels"] = transformed_labels

    # Rearrange df Image-Text columns for better visualization

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
        ]
    ]

    df.to_feather(f"{DATA_DIR}/annotations_region_final.feather")
    print(f"Dataframe saved to {DATA_DIR}")


if __name__ == "__main__":
    main()

# import os
# import sys

# import cv2

# sys.path.append("C:/Users/DryLab/Desktop/ViLLA/mimic-cxr")

# import numpy as np
# import pandas as pd
# import scipy.sparse as sp
# import torch
# from models.HybridGNet2IGSC import Hybrid
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from tqdm.auto import tqdm
# from utils.utils import genMatrixesLungsHeart, scipy_to_torch_sparse

# DATA_DIR = "C:/Users/DryLab/Desktop/villa/data/open-i"


# def extract_id(path):
#     basename = os.path.basename(path)

#     # Extract the ID from the filename
#     return "_".join(basename.split("_")).replace(".txt", "").replace(".png", "")


# def loadModel(device):
#     A, AD, D, U = genMatrixesLungsHeart()
#     N1 = A.shape[0]
#     N2 = AD.shape[0]

#     A = sp.csc_matrix(A).tocoo()
#     AD = sp.csc_matrix(AD).tocoo()
#     D = sp.csc_matrix(D).tocoo()
#     U = sp.csc_matrix(U).tocoo()

#     D_ = [D.copy()]
#     U_ = [U.copy()]

#     config = {}

#     config["n_nodes"] = [N1, N1, N1, N2, N2, N2]
#     A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]

#     A_t, D_t, U_t = (
#         [scipy_to_torch_sparse(x).to(device) for x in X] for X in (A_, D_, U_)
#     )

#     config["latents"] = 64
#     config["inputsize"] = 1024

#     f = 32
#     config["filters"] = [2, f, f, f, f // 2, f // 2, f // 2]
#     config["skip_features"] = f

#     hybrid = Hybrid(config.copy(), D_t, U_t, A_t).to(device)
#     hybrid.load_state_dict(
#         torch.load(
#             "C:/Users/DryLab/Desktop/ViLLA/mimic-cxr/weights/weights.pt",
#             map_location=torch.device(device),
#         )
#     )
#     hybrid.eval()

#     return hybrid


# def getMasks(landmarks, h, w):
#     RL = landmarks[0:44]
#     LL = landmarks[44:94]
#     H = landmarks[94:]

#     RL = RL.reshape(-1, 1, 2).astype("int")
#     LL = LL.reshape(-1, 1, 2).astype("int")
#     H = H.reshape(-1, 1, 2).astype("int")

#     RL_mask = np.zeros([h, w], dtype="uint8")
#     LL_mask = np.zeros([h, w], dtype="uint8")
#     H_mask = np.zeros([h, w], dtype="uint8")

#     RL_mask = cv2.drawContours(RL_mask, [RL], -1, 255, -1)
#     LL_mask = cv2.drawContours(LL_mask, [LL], -1, 255, -1)
#     H_mask = cv2.drawContours(H_mask, [H], -1, 255, -1)

#     return RL_mask, LL_mask, H_mask


# def removePreprocess(output, info):
#     h, w, padding = info
#     if h.size != 1024 or w.size != 1024:
#         output = output * h
#     else:
#         output = output * 1024
#     padh, padw, auxh, auxw = padding
#     output[:, 0] = output[:, 0] - padw // 2
#     output[:, 1] = output[:, 1] - padh // 2
#     return output


# def get_bounding_box(region_coords):
#     # Find the indices where the region_label exists
#     indices = np.nonzero(region_coords)

#     # If there are no coordinates for the region_label, return a placeholder
#     if len(indices[0]) == 0:
#         return np.array([None, None, None, None])  # Placeholder for empty regions

#     # Get the min and max x and y coordinates
#     min_y, min_x = np.min(indices, axis=1)
#     max_y, max_x = np.max(indices, axis=1)

#     return np.array([min_x, min_y, max_x, max_y])


# def process_row(region_coords, original_shape):
#     # Reshape the region_coords to the original shape
#     right_lung_coords = region_coords[0]  # Right Lung
#     left_lung_coords = region_coords[1]  # Left Lung
#     heart_coords = region_coords[2]  # Heart

#     right_lung_2d = right_lung_coords.reshape(original_shape)
#     left_lung_2d = left_lung_coords.reshape(original_shape)
#     heart_2d = heart_coords.reshape(original_shape)

#     # Calculate the bounding box for each region
#     left_lung_bbox = get_bounding_box(left_lung_2d)
#     right_lung_bbox = get_bounding_box(right_lung_2d)
#     heart_bbox = get_bounding_box(heart_2d)

#     # Concatenate all bounding boxes into a single 1D array
#     region_bbox = [
#         np.array(right_lung_bbox.reshape(-1)),
#         np.array(left_lung_bbox.reshape(-1)),
#         np.array(heart_bbox.reshape(-1)),
#     ]

#     # Initialize a NumPy array to store the reg_coords for this image
#     reg_bbox_array = np.empty(3, dtype=object)

#     # Store each organ's coordinates in the reg_coords_array
#     for j in range(3):
#         reg_bbox_array[j] = region_bbox[j]

#     return reg_bbox_array


# class CustomDataset(Dataset):
#     def __init__(self, image_paths, device, transform=None):
#         self.image_paths = image_paths
#         self.transform = transform
#         self.device = device  # Store device information

#     def __len__(self):
#         return len(self.image_paths)

#     def pad_to_square(self, img):
#         h, w = img.shape[:2]

#         padh, padw, auxh, auxw = 0, 0, 0, 0  # Initialize padding variables

#         if h > w:
#             padw = h - w
#             auxw = padw % 2
#             img = np.pad(img, ((0, 0), (padw // 2, padw // 2 + auxw)), "constant")
#         else:
#             padh = w - h
#             auxh = padh % 2
#             img = np.pad(img, ((padh // 2, padh // 2 + auxh), (0, 0)), "constant")

#         return img, (padh, padw, auxh, auxw)

#     def preprocess(self, input_img):
#         img, padding = self.pad_to_square(input_img)

#         h, w = img.shape[:2]
#         if h != 1024 or w != 1024:
#             img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)

#         return img, (h, w, padding)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = cv2.imread(image_path, 0) / 255.0

#         # Convert image to float32
#         image = image.astype(np.float32)

#         # image = Image.open(image_path).convert("L")
#         # image = np.array(image, dtype=np.float32) / 255.0
#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # image_tensor = torch.from_numpy(img).unsqueeze(0).float()
#         image, (h, w, padding) = self.preprocess(image)
#         if self.transform:
#             image = self.transform(image)

#         return image, (h, w, padding)


# def generate_segmentation_mask(model, dataloader, device):
#     print(f"Segmenting images using {device}")
#     model.eval()
#     segmentation_results = []

#     with torch.no_grad():
#         for images, (h, w, padding) in tqdm(dataloader):
#             images = images.to(device)
#             model = model.to(device)

#             # Process the model output for the entire batch
#             outputs = model(images)[0]  # Shape: (batch_size, num_points, 2)

#             # Loop through the batch to generate masks for each image
#             for batch_idx in range(images.shape[0]):
#                 # Get height, width, and padding for the current image
#                 height, width, pad = (
#                     h[batch_idx],
#                     w[batch_idx],
#                     (
#                         padding[0][batch_idx],
#                         padding[1][batch_idx],
#                         padding[2][batch_idx],
#                         padding[3][batch_idx],
#                     ),
#                 )
#                 # print(height, width, pad)

#                 # Remove preprocessing from the output for the current image
#                 output = removePreprocess(outputs[batch_idx], (height, width, pad))
#                 output = output.cpu().numpy().astype("int")

#                 # Extract region points
#                 RL = output[:44]
#                 LL = output[44:94]
#                 H = output[94:]

#                 RL = RL.reshape(-1, 1, 2).astype("int")
#                 LL = LL.reshape(-1, 1, 2).astype("int")
#                 H = H.reshape(-1, 1, 2).astype("int")

#                 # Create masks for each region
#                 RL_mask = np.zeros([height, width], dtype="uint8")
#                 LL_mask = np.zeros([height, width], dtype="uint8")
#                 H_mask = np.zeros([height, width], dtype="uint8")

#                 RL_mask = cv2.drawContours(RL_mask, [RL], -1, 255, -1)
#                 LL_mask = cv2.drawContours(LL_mask, [LL], -1, 255, -1)
#                 H_mask = cv2.drawContours(H_mask, [H], -1, 255, -1)

#                 # # Resize masks to 224x224
#                 # RL_mask = cv2.resize(
#                 #     RL_mask, (224, 224), interpolation=cv2.INTER_NEAREST
#                 # )
#                 # LL_mask = cv2.resize(
#                 #     LL_mask, (224, 224), interpolation=cv2.INTER_NEAREST
#                 # )
#                 # H_mask = cv2.resize(H_mask, (224, 224), interpolation=cv2.INTER_NEAREST)

#                 RL_mask = np.array(RL_mask)
#                 LL_mask = np.array(LL_mask)
#                 H_mask = np.array(H_mask)

#                 segmentation_results.append((RL_mask, LL_mask, H_mask))

#     return segmentation_results


# def resize_bounding_box(bbox, orig_width, orig_height, target_size=224):
#     """
#     Resizes bounding box coordinates to fit a target image size.

#     Parameters:
#     bbox (tuple): Original bounding box coordinates as (x_min, y_min, x_max, y_max).
#     orig_width (int): Width of the original image.
#     orig_height (int): Height of the original image.
#     target_size (int): Target dimension (width and height) for the resized image.

#     Returns:
#     tuple: Resized bounding box coordinates (x_min_resized, y_min_resized, x_max_resized, y_max_resized).
#     """
#     x_min, y_min, x_max, y_max = bbox

#     # Calculate scaling factors
#     scale_x = target_size / orig_width
#     scale_y = target_size / orig_height

#     # Scale coordinates
#     x_min_resized = int(x_min * scale_x)
#     y_min_resized = int(y_min * scale_y)
#     x_max_resized = int(x_max * scale_x)
#     y_max_resized = int(y_max * scale_y)

#     return (x_min_resized, y_min_resized, x_max_resized, y_max_resized)


# def main():
#     # Setup device agnostic code
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     image_dir = "C:/Users/DryLab/Desktop/ConVIRT/Dataset/open-i/data/front"
#     image_paths = sorted(
#         [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)],
#         key=lambda x: int(os.path.basename(x).split("CXR")[-1].split("_")[0]),
#     )
#     text_dir = "C:/Users/DryLab/Desktop/ConVIRT/Dataset/open-i/data/ecgen-radiology-txt"

#     text_paths = sorted(
#         os.listdir(text_dir),
#         key=lambda filename: int(filename.split("_")[0].replace("CXR", "")),
#     )

#     text_paths = [os.path.join(text_dir, path) for path in text_paths]

#     # Create dictionaries to map image IDs to text paths and vice versa
#     text_dict = {extract_id(path): path for path in text_paths}
#     image_dict = {extract_id(path): path for path in image_paths}

#     common_ids = set(image_dict.keys()) & set(text_dict.keys())

#     paired_data = []
#     for common_id in common_ids:
#         paired_data.append(
#             {
#                 "image_filepath": image_dict[common_id],
#                 "text_filepath": text_dict[common_id],
#             }
#         )

#     paired_data.sort(key=lambda x: x["image_filepath"])

#     df = pd.DataFrame(paired_data)
#     df["image_id"] = range(len(df))

#     image_paths_list = df["image_filepath"].tolist()

#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
#     )

#     train_dataset = CustomDataset(image_paths_list, device, transform=transform)
#     train_dataloader = DataLoader(
#         train_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
#     )

#     model = loadModel(device)

#     full_mask = generate_segmentation_mask(model, train_dataloader, device)

#     df["image_size"] = [list(mask[0].shape) for mask in full_mask]

#     # Initialize a NumPy array to store reg_coords for all images
#     all_reg_coords = np.empty(len(full_mask), dtype=object)

#     for i in tqdm(range(len(full_mask))):
#         mask = np.array(full_mask[i])

#         # Convert mask to torch tensor for processing
#         mask_tensor = torch.from_numpy(mask)

#         # Create boolean masks for each organ
#         right_lung_mask = mask_tensor[0]
#         left_lung_mask = mask_tensor[1]
#         heart_mask = mask_tensor[2]

#         # Combine the coordinates into an array of arrays
#         reg_coords = [
#             np.array(right_lung_mask.cpu().numpy().astype(np.uint64).reshape(-1)),
#             np.array(left_lung_mask.cpu().numpy().astype(np.uint64).reshape(-1)),
#             np.array(heart_mask.cpu().numpy().astype(np.uint64).reshape(-1)),
#         ]

#         # Initialize a NumPy array to store the reg_coords for this image
#         reg_coords_array = np.empty(3, dtype=object)

#         # Store each organ's coordinates in the reg_coords_array
#         for j in range(3):
#             reg_coords_array[j] = reg_coords[j]

#         # Append to the main NumPy array
#         all_reg_coords[i] = reg_coords_array

#     df["region_coord"] = all_reg_coords

#     df["num_regions"] = df["region_coord"].apply(lambda x: len(x))

#     # Initialize a NumPy array to store region_bbox for all images
#     all_region_bbox = np.empty(len(all_reg_coords), dtype=object)

#     for i in tqdm(range(len(all_reg_coords))):
#         region_coords = all_reg_coords[i]

#         # Assuming original_shape is known and is the same for all images
#         # Replace `original_shape` with the actual shape of the mask
#         original_shape = df["image_size"].iloc[i]

#         # Process each row to compute bounding boxes
#         region_bbox = process_row(region_coords, original_shape)

#         # Resize masks to 224x224
#         for j in range(3):
#             region_bbox[j] = resize_bounding_box(
#                 region_bbox[j], original_shape[1], original_shape[0]
#             )

#         # Store the result in the NumPy array
#         all_region_bbox[i] = region_bbox

#         df["region_bbox"] = all_region_bbox

#     # Add region labels to the DataFrame
#     df["region_labels"] = [np.array([1, 2, 3]) for _ in range(len(df))]
#     df["region_labels"] = df["region_labels"].apply(lambda labels: [labels])

#     # Initialize an empty NumPy array to store the transformed region labels
#     transformed_labels = np.empty(len(df), dtype=object)

#     # Process each row in the DataFrame
#     for i in tqdm(range(len(df))):
#         # Extract the nested array
#         nested_array = df["region_labels"].iloc[i][0]

#         # Initialize an empty NumPy array to store the transformed labels
#         labels_array = np.empty(len(nested_array), dtype=object)

#         # Convert each label to its own NumPy array and store it
#         for j, label in enumerate(nested_array):
#             labels_array[j] = np.array([label], dtype=np.uint64)

#         # Store the transformed labels array in the main array
#         transformed_labels[i] = labels_array

#     # Update the DataFrame with the transformed labels
#     df["region_labels"] = transformed_labels

#     # Drop region_coord column
#     df.drop("region_coord", axis=1, inplace=True)

#     # Rearrange df Image-Text columns for better visualization
#     df = df[
#         [
#             "image_id",
#             "image_size",
#             "image_filepath",
#             # "region_coord",
#             "region_bbox",
#             "region_labels",
#             "num_regions",
#             "text_filepath",
#         ]
#     ]

#     df.to_feather(f"{DATA_DIR}/annotations_region_final.feather")
#     print(f"Annotations saved to {DATA_DIR}")


# if __name__ == "__main__":
#     main()
