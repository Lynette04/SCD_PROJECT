import math #for the rotation angle
import os #to count files in folders

import numpy as np
from PIL import Image #to open the image files
import torch #module that can teach AI
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

img_size = (224,224) #224 pixels tall and 224 pixels wide
batch_size = 16 #16 images in a batch

#Each image batch has the shape (batch, 3, height, width) because PyTorch keeps the colours first
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1) * 255.0
IMAGENET_STD = torch.tensor([0.229,0.224,0.225]).reshape(3, 1, 1) * 255.0

#The normalization for MobileNetV2: pixel values of 0 to 255 become -1 to 1
def preprocess_input(images):
    return images / 127.5 - 1.0

def efficientnet_preprocess(images):
    return (images - IMAGENET_MEAN) / IMAGENET_STD


#The 4 layers below follow the formulas of the Keras layers they replace.
#The built-in PyTorch versions work differently, for example they fill the corners of a rotated image with black.

class RandomFlip(nn.Module): #turn the images in opposite directions like right/left and up/down
    def forward(self, images):
        #one decision per image, used for both directions just like the Keras "horizontal_and_vertical" mode
        flips = torch.rand(len(images), 1, 1, 1) <= 0.5
        images = torch.where(flips, images.flip(3), images)
        images = torch.where(flips, images.flip(2), images)
        return images

class RandomRotation(nn.Module): #rotate the image by up to 10% of a full turn either way
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, images):
        count, _, height, width = images.shape
        angles = (torch.rand(count) * 2 * self.factor - self.factor) * 2 * math.pi
        cos, sin = angles.cos().reshape(-1, 1, 1), angles.sin().reshape(-1, 1, 1)
        cx, cy = (width - 1) / 2, (height - 1) / 2

        #for every pixel of the new image, find which pixel of the old image it comes from
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        source_x = cos * (x - cx) - sin * (y - cy) + cx
        source_y = sin * (x - cx) + cos * (y - cy) + cy

        #grid_sample expects positions between -1 and 1. Points outside are filled by reflecting the image.
        grid = torch.stack(
            [(2 * source_x + 1) / width - 1, (2 * source_y + 1) / height - 1], dim=-1
        )
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

class RandomBrightness(nn.Module): #randomize the brightness by 0.1
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, images):
        change = (torch.rand(len(images), 1, 1, 1) * 2 * self.factor - self.factor) * 255.0
        return (images + change).clamp(0, 255)

class RandomContrast(nn.Module): #randomize the contrast by 0.1
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, images):
        contrast = 1.0 - self.factor + torch.rand(len(images), 1, 1, 1) * 2 * self.factor
        mean = images.mean(dim=(2, 3), keepdim=True)
        return ((images - mean) * contrast + mean).clamp(0, 255)


#create a sort of pipeline with ordered steps(sequential) to perform on the training set images
data_augmentation = nn.Sequential(
    RandomFlip(),
    RandomRotation(0.5),
    RandomBrightness(0.1),
    RandomContrast(0.1),
)


#Reads the images in a folder that has one sub folder per class.
#The classes and the files are taken in alphabetical order, so 0 is normal and 1 is sickle.
class ImageFolder(Dataset):
    def __init__(self, folder):
        self.class_names = sorted(os.listdir(folder))
        self.samples = []
        for label, class_name in enumerate(self.class_names):
            class_folder = os.path.join(folder, class_name)
            for filename in sorted(os.listdir(class_folder)):
                self.samples.append((os.path.join(class_folder, filename), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.float32)
        image = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        #resize to 224 by 224 pixels, the same way Keras does
        image = F.interpolate(image, size=img_size, mode="bilinear", align_corners=False, antialias=False)
        return image.squeeze(0), torch.tensor([label], dtype=torch.float32)


#Groups the images into batches and applies a function to each batch, like dataset.map in TensorFlow
class BatchedDataset:
    def __init__(self, folder, prepare_batch, shuffle):
        self.images = ImageFolder(folder)
        self.class_names = self.images.class_names
        self.prepare_batch = prepare_batch
        self.loader = DataLoader(self.images, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for images, labels in self.loader:
            yield self.prepare_batch(images), labels


def get_datasets(preprocess_fn=preprocess_input):
    #Function that takes a batch of images, augments then normalizes them.
    def prepare_train_batch(images):
        images = data_augmentation(images)
        images = preprocess_fn(images)
        return images
    #Similar function to above but has no augmentation since the other images are not for training the model
    def prepare_evaluation_batch(images):
        images = preprocess_fn(images)
        return images

    #check in the train folder inside data and resize each image to 224 by 224 pixels
    #and group the images into groups of 16 called batches.
    #The labels are in binary 0 for Normal and 1 for sickle
    training_dataset = BatchedDataset("data/train", prepare_train_batch, shuffle=True)
    #the same is done for the val and test folders but the files aren't shuffled since the order doesnt matter for evaluation.
    validation_dataset = BatchedDataset("data/val", prepare_evaluation_batch, shuffle=False)
    test_dataset = BatchedDataset("data/test", prepare_evaluation_batch, shuffle=False)

    #Store the class names of the training dataset in its own variable claas_names
    class_names = training_dataset.class_names

    return training_dataset, validation_dataset, test_dataset, class_names

def get_class_weights(): #function to calculate class weights
    train_path = "data/train" #path to count files
    class_counts = {} #dictionary to store the number of images in each class of normal and sickle.

    #for loop to count the number of files in each class and add then add the number to the dictionary of class counts.
    #the loop has to get the class names in alphabetical order(normal then sickle) to match the order of 0-normal and 1-sickle
    for class_name in sorted(os.listdir(train_path)):
        class_folder = os.path.join(train_path, class_name)
        class_counts[class_name] = len(os.listdir(class_folder))



    #to solve the imbalance of normal images being fewer than the sickle images
    #let's introduce class weights so that each class is multiplied by its weight.
    #We don't want the model to become lazy and wrongly classify normal images as sickle images. It has to pay more attention to the normal since they are the minority in number.

    total_number_of_images = sum(class_counts.values()) #sum of images in normal and sickle folders
    number_of_classes = len(class_counts) # 2 in number: normal and sickle

    class_weights= {} #dictionary to hold the weights for each class
    sorted_class_names = sorted(class_counts.keys()) #alphabetical order for the classes

    for index, class_name in enumerate(sorted_class_names): #enumerate gives both the index and the item in that position.
        #class with fewer images gets a bigger weight so mistakes are amplified and model has to work on reducing mistakes
        class_weights[index] = total_number_of_images / (number_of_classes * class_counts[class_name])

    return class_weights, class_counts

#Testing block of code to see that the functions work.
if __name__ == "__main__":
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets()
    print("Class names:", class_names)

    class_weights, class_counts = get_class_weights()
    print("Training image counts:", class_counts)
    print("Class weights:", class_weights)

    #get one batch and split in into images and the matching labels
    images, labels = next(iter(training_dataset))
    print("Shape of one batch of images:", images.shape) #Expected output is (16,3,224,224) 3: each pixel has 3 color values RGB
    print("Pixel value raange after preprocessing:", float(images.min()), "to", float(images.max()))
