#Import the necessities for the program to run such as tensorflow, keras_hub and layers of the model.
import tensorflow as tf 
import keras_hub
from tensorflow.keras import layers #layers to help uc check the type of each layer.

#Load the frozen efficientnet model that was trained earlier
model = tf.keras.models.load_model("sickle_cell_effnet_lite0.keras")
#Position 1 is the EfficientNet backbone 
base_model = model.layers[1]

#Count the number of layers in the backbone
n = len(base_model.layers)
print("Backbone layers: ", n)

#Count the batchnormalization layers within the backbone layer's list.
print("Top-level BatchNormalisation layers:", sum(isinstance(lr,layers.BatchNormalization) for lr in base_model.layers))

#Count layers that have smaller layers within them
print("Layers that contain sublayers:", sum(hasattr(lr, "layers") for lr in base_model.layers))

#Display the last 35 layers: their positions, types, names and number of parameters
for i, lr in enumerate(base_model.layers[-35:], start=n - 35):
    print(i, type(lr).__name__, lr.name, lr.count_params())