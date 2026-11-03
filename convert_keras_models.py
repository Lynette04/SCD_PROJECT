#Converts the models trained with Keras into PyTorch files, without retraining them.
#This only needs to run once. The MobileNetV2 .pt files are already in the project.
#Any .keras file that is missing on this computer is skipped.
import io #to open the weights file from inside the .keras file without unpacking it
import os #to check whether a file exists
import zipfile #a .keras file is really a zip file

import h5py #module to read the weights
import torch

from pytorch_mobilenet import load_keras_base_weights
from pytorch_training import SickleCellClassifier, save_model

MOBILENET_GROUP = "layers/functional/layers"
EFFICIENTNET_GROUP = "layers/efficient_net_backbone/layers"

#The Keras file, the PyTorch file to create, the architecture, its alpha value and where its weights are kept
MODELS = [
    ("sickle_cell_model.keras", "sickle_cell_model.pt", "mobilenet_v2", 1.0, MOBILENET_GROUP),
    ("Sickle_cell_fine_tuned-mode.keras", "Sickle_cell_fine_tuned-mode.pt", "mobilenet_v2", 1.0, MOBILENET_GROUP),
    ("sickle_cell_edge_model.keras", "sickle_cell_edge_model.pt", "mobilenet_v2", 0.35, MOBILENET_GROUP),
    ("sickle_cell_effnet_lite0.keras", "sickle_cell_effnet_lite0.pt", "efficientnet_lite0", 1.0, EFFICIENTNET_GROUP),
    ("sickle_cell_effnet_lite0_ft30.keras", "sickle_cell_effnet_lite0_ft30.pt", "efficientnet_lite0", 1.0, EFFICIENTNET_GROUP),
]


def convert(keras_file, pytorch_file, architecture, alpha, group):
    model = SickleCellClassifier(architecture, alpha)

    with zipfile.ZipFile(keras_file) as archive:
        weights_file = io.BytesIO(archive.read("model.weights.h5"))

    with h5py.File(weights_file, "r") as weights:
        load_keras_base_weights(model.base_model, weights[group])
        #The Dense kernel is (inputs, outputs) in Keras and (outputs, inputs) in PyTorch
        with torch.no_grad():
            model.dense.weight.copy_(torch.from_numpy(weights["layers/dense/vars/0"][()].T))
            model.dense.bias.copy_(torch.from_numpy(weights["layers/dense/vars/1"][()]))

    save_model(model, pytorch_file)
    print(f"{keras_file} converted to {pytorch_file}")


if __name__ == "__main__":
    for keras_file, *details in MODELS:
        if os.path.exists(keras_file):
            convert(keras_file, *details)
        else:
            print(f"{keras_file} was not found, skipping it")
