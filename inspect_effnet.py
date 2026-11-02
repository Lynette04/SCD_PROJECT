#Import the necessities for the program to run such as the BatchNorm layer type and the model tools.
from torch import nn
from pytorch_training import count_params, load_model

#Load the frozen efficientnet model that was trained earlier
model = load_model("sickle_cell_effnet_lite0.pt")
#The EfficientNet backbone
base_model = model.base_model

#Count the number of layers in the backbone
n = len(base_model.layers)
print("Backbone layers: ", n)

#Count the batchnormalization layers within the backbone layer's list.
print("Top-level BatchNormalisation layers:", sum(isinstance(lr,nn.BatchNorm2d) for lr in base_model.layers))

#Count layers that have smaller layers within them
print("Layers that contain sublayers:", sum(hasattr(lr, "layers") for lr in base_model.layers))

#Display the last 35 layers: their positions, types, names and number of parameters
for i, (name, lr) in enumerate(list(base_model.named_layers.items())[-35:], start=n - 35):
    print(i, type(lr).__name__, name, count_params(lr))
