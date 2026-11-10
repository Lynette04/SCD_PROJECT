import math
import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset 

img_size = (224, 224)
batch_size = 32

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1) * 255.0
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1) * 255.0

def efficientnet_preprocess(images):
    return (images - IMAGENET_MEAN) / IMAGENET_STD 

class RandomFlip(nn.Module):
    def forward(self, images):
        flips = torch.rand(len(images), 1, 1, 1) <= 0.5
        images = torch.where(flips, images.flip(3), images)
        images = torch.where(flips, images.flip(2), images)
        return images

class RandomRotation(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, images):
        count, _, height, width = images.shape
        angles = (torch.rand(count) * 2 * self.factor - self.factor) * 2 * math.pi
        cos, sin = angles.cos().reshape(-1, 1, 1), angles.sin().reshape(-1, 1, 1)
        cx, cy = (width - 1) / 2, (height - 1) / 2

        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        source_x = cos * (x - cx) - sin * (y - cy) + cx
        source_y = sin * (x - cx) + cos * (y - cy) + cy

        grid = torch.stack(
            [(2 * source_x + 1) / width - 1, (2 * source_y + 1) / height - 1], dim=-1
        )
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

class RandomBrightness(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, images):
        change = (torch.rand(len(images), 1, 1, 1) * 2 * self.factor - self.factor) * 255.0
        return (images + change).clamp(0, 255)

class RandomContrast(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, images):
        contrast = 1.0 - self.factor + torch.rand(len(images), 1, 1, 1) * 2 * self.factor
        mean = images.mean(dim=(2, 3), keepdim=True)
        return ((images - mean) * contrast + mean).clamp(0, 255)

data_augmentation = nn.Sequential(
    RandomFlip(),
    RandomRotation(0.5),
    RandomBrightness(0.1),
    RandomContrast(0.1),
)    

class SKDrepaDataset(Dataset):
    def __init__(self, split_folder):
        # We focus on the two classes: normal and sickle
        self.classes = {"normale_cell": 0, "sickle_cell": 1}
        self.samples = []
        for class_name, label in self.classes.items():
            folder = os.path.join(split_folder, class_name)
            if os.path.isdir(folder):
                for filename in sorted(os.listdir(folder)):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(folder, filename), label))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.float32)
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        tensor = F.interpolate(tensor, size=img_size, mode="bilinear", align_corners=False)
        return tensor.squeeze(0), torch.tensor([label], dtype=torch.float32)                    
    
class BatchedDataset:
    def __init__(self, split_folder, prepare_batch, shuffle):
        self.dataset = SKDrepaDataset(split_folder)
        self.prepare_batch = prepare_batch
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for images, labels in self.loader:
            yield self.prepare_batch(images), labels

def _resolve_data_root(data_root=None):
    if data_root and os.path.isdir(data_root):
        return data_root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        data_root if data_root else "",
        r"..\SKDrepaData-v2-clean",
        r"..\..\SKDrepaData-v2-clean",
        os.path.join(script_dir, "..", "..", "SKDrepaData-v2-clean"),
        r"C:\Users\kaboh\OneDrive\Desktop\Project-prototype\SKDrepaData-v2-clean",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return r"..\..\SKDrepaData-v2-clean"

def get_datasets(data_root=None, preprocess_fn=efficientnet_preprocess):
    data_root = _resolve_data_root(data_root)
    def prepare_train_batch(images):
        images = data_augmentation(images)
        return preprocess_fn(images)

    def prepare_eval_batch(images):
        return preprocess_fn(images)
    
    training_dataset = BatchedDataset(os.path.join(data_root,"train"), prepare_train_batch, shuffle=True)
    validation_dataset = BatchedDataset(os.path.join(data_root, "val"), prepare_eval_batch, shuffle=False)
    test_dataset = BatchedDataset(os.path.join(data_root, "test"), prepare_eval_batch, shuffle=False)

    class_names = ["normale_cell", "sickle_cell"]
    return training_dataset, validation_dataset, test_dataset, class_names
    
def get_class_weights(data_root=None):
        data_root = _resolve_data_root(data_root)
        train_dir = os.path.join(data_root, "train")
        counts = {
            0: len(os.listdir(os.path.join(train_dir, "normale_cell"))),
            1: len(os.listdir(os.path.join(train_dir, "sickle_cell"))),
        }
        total = sum(counts.values())
        weights = {cls: total / (2.0 * count) for cls, count in counts.items()}
        return weights, counts  

if __name__ == "__main__":
    train_ds, val_ds, test_ds, names = get_datasets()
    weights, counts = get_class_weights()
    print("Class names:", names)
    print("Training counts (Normal, Sickle):", counts)
    print("Class weights:", weights)
    print("Total training batches:", len(train_ds))      