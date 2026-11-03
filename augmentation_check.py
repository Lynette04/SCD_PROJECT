"""
augmentation_check.py

Checks whether the augmentation pipeline in step2_data_pipeline.py actually changes images.
In Keras the augmentation layers only worked when called with training=True.
The PyTorch versions don't depend on a training flag, so they should change the image
whether the model is in training mode or in evaluation mode.
Read-only: it only prints results.

Run from the project folder (the one containing step2_data_pipeline.py):
    python augmentation_check.py
"""
import torch

from step2_data_pipeline import data_augmentation

x = torch.rand(1, 3, 224, 224) * 255 #one random image with pixel values between 0 and 255, colour channel first


def difference_between_two_augmentations():
    return float((data_augmentation(x) - data_augmentation(x)).abs().mean())


data_augmentation.train()
in_training_mode = difference_between_two_augmentations()
data_augmentation.eval()
in_evaluation_mode = difference_between_two_augmentations()
data_augmentation.train()

print("Difference between two augmentations of the same image")
print(f"  in training mode (as called in the pipeline): {in_training_mode:.4f}")
print(f"  in evaluation mode:                           {in_evaluation_mode:.4f}")

if in_training_mode == 0.0:
    print("\nAugmentation is NOT active as currently called.")
else:
    print("\nAugmentation is active as currently called.")
