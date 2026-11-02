"""
check_augmentation.py
 
Checks whether the augmentation pipeline in step2_data_pipeline.py actually changes images
when it is called the way the pipeline calls it (without training=True).
Read-only: it only prints results.
 
Run from the project folder (the one containing step2_data_pipeline.py):
    python check_augmentation.py
"""
import tensorflow as tf
 
from step2_data_pipeline import data_augmentation
 
x = tf.random.uniform((1, 224, 224, 3), 0, 255)
 
as_called_in_pipeline = float(tf.reduce_mean(tf.abs(data_augmentation(x) - data_augmentation(x))))
with_training_true = float(
    tf.reduce_mean(tf.abs(data_augmentation(x, training=True) - data_augmentation(x, training=True)))
)
 
print("Difference between two augmentations of the same image")
print(f"  called as in the pipeline (no training argument): {as_called_in_pipeline:.4f}")
print(f"  called with training=True:                        {with_training_true:.4f}")
 
if as_called_in_pipeline == 0.0:
    print("\nAugmentation is NOT active as currently called. Use data_augmentation(images, training=True).")
else:
    print("\nAugmentation is active as currently called.")