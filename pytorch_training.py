import copy #to keep a safe copy of the best weights during training
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

import pytorch_efficientnet
import pytorch_mobilenet


#Same as tf.keras.utils.set_random_seed, sets the seed for every source of randomness used here
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#Base model + GlobalAveragePooling + Dropout + Dense(1, sigmoid), the same head on every model
class SickleCellClassifier(nn.Module):
    def __init__(self, architecture="mobilenet_v2", alpha=1.0):
        super().__init__()
        self.architecture = architecture
        self.alpha = alpha
        if architecture == "mobilenet_v2":
            self.base_model = pytorch_mobilenet.MobileNetV2(alpha=alpha)
        elif architecture == "efficientnet_lite0":
            self.base_model = pytorch_efficientnet.EfficientNetLite0()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.dropout = nn.Dropout(0.3) #same 30% of the values randomly switched off during training
        self.dense = nn.Linear(1280, 1)
        nn.init.xavier_uniform_(self.dense.weight) #the starting values Keras gives a Dense layer
        nn.init.zeros_(self.dense.bias)

        freeze_base_model(self)
        self.base_model.eval()

    def forward(self, images):
        x = self.base_model(images)
        x = x.mean(dim=(2, 3)) #GlobalAveragePooling2D: squeeze each image grid into a compact list of numbers
        x = self.dropout(x)
        return torch.sigmoid(self.dense(x)) #one number between 0 and 1 (0 is normal and 1 is sickle)

    #The original models call base_model(inputs, training=False), so the base model always
    #stays in inference mode, even while fine tuning. The BatchNormalization layers rely on
    #this: in PyTorch, freezing them is not enough to stop them from updating.
    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self


#Builds a new model that starts from the ImageNet weights
def build_pretrained_model(architecture="mobilenet_v2", alpha=1.0):
    model = SickleCellClassifier(architecture, alpha)
    if architecture == "mobilenet_v2":
        pytorch_mobilenet.load_imagenet_weights(model.base_model)
    else:
        pytorch_efficientnet.load_imagenet_weights(model.base_model)
    return model


#Like setting layer.trainable in Keras. The flag is also kept for layers with no weights
#so that the "layers actually unfrozen" count is the same as the one Keras prints.
def set_layer_trainable(layer, trainable):
    layer.trainable = trainable
    for parameter in layer.parameters():
        parameter.requires_grad = trainable


def freeze_base_model(model):
    for layer in model.base_model.layers:
        set_layer_trainable(layer, False)


#Switches on the last few layers of the base model, leaving BatchNormalization frozen
def unfreeze_last_layers(model, number_of_layers):
    base_layers = model.base_model.layers
    first_unfrozen = len(base_layers) - number_of_layers
    for i, layer in enumerate(base_layers):
        if i < first_unfrozen:
            set_layer_trainable(layer, False) #early layers keep their ImageNet knowledge intact
        elif isinstance(layer, nn.BatchNorm2d):
            set_layer_trainable(layer, False)
        else:
            set_layer_trainable(layer, True) #later layers are allowed to adjust to blood cell images
    return sum(1 for layer in base_layers if layer.trainable)


#Same count as Keras: weights plus the moving averages stored in BatchNormalization
def count_params(module):
    weights = sum(p.numel() for p in module.parameters())
    moving_averages = sum(
        b.numel() for name, b in module.named_buffers() if not name.endswith("num_batches_tracked")
    )
    return weights + moving_averages


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


#Prints a table of the layers in the model, like model.summary() in Keras
def summary(model):
    print(f"{'Layer':<34}{'Type':<16}{'Params':>12}")
    print("-" * 62)
    for name, layer in model.base_model.named_layers.items():
        print(f"{name:<34}{type(layer).__name__:<16}{count_params(layer):>12,}")
    print(f"{'dropout':<34}{'Dropout':<16}{0:>12,}")
    print(f"{'dense':<34}{'Linear':<16}{count_params(model.dense):>12,}")
    print("-" * 62)
    print(f"Total params: {count_params(model):,}")
    print(f"Trainable params: {count_trainable_params(model):,}")


#Saves the entire model details to a single file to prevent retraining
def save_model(model, path):
    torch.save(
        {
            "architecture": model.architecture,
            "alpha": model.alpha,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(path):
    checkpoint = torch.load(path, weights_only=True)
    model = SickleCellClassifier(checkpoint["architecture"], checkpoint["alpha"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


#The binary_crossentropy formula of Keras, including the way it keeps values away from 0 and 1
def binary_crossentropy(probabilities, labels):
    probabilities = probabilities.clamp(1e-7, 1 - 1e-7)
    return -(labels * probabilities.log() + (1 - labels) * (1 - probabilities).log())


#Keeps count of right and wrong guesses to work out accuracy, precision and recall
@dataclass
class Metrics:
    loss_sum: float = 0.0
    images: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def update(self, losses, probabilities, labels):
        predictions = probabilities > 0.5
        positives = labels > 0.5
        self.loss_sum += losses.sum().item()
        self.images += len(labels)
        self.true_positives += (predictions & positives).sum().item()
        self.true_negatives += (~predictions & ~positives).sum().item()
        self.false_positives += (predictions & ~positives).sum().item()
        self.false_negatives += (~predictions & positives).sum().item()

    def results(self):
        guessed_sickle = self.true_positives + self.false_positives
        actually_sickle = self.true_positives + self.false_negatives
        return {
            "accuracy": (self.true_positives + self.true_negatives) / self.images,
            "loss": self.loss_sum / self.images,
            "precision": self.true_positives / guessed_sickle if guessed_sickle else 0.0,
            "recall": self.true_positives / actually_sickle if actually_sickle else 0.0,
        }


#Like model.evaluate(dataset, return_dict=True) in Keras
@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    metrics = Metrics()
    for images, labels in dataset:
        probabilities = model(images)
        metrics.update(binary_crossentropy(probabilities, labels), probabilities, labels)
    return metrics.results()


#Like model.predict(dataset) in Keras, gives the confidence (0 to 1) for every image
@torch.no_grad()
def predict(model, dataset):
    model.eval()
    return torch.cat([model(images) for images, _ in dataset]).numpy().ravel()


#Like model.fit(...) with the Adam optimizer, class weights and an EarlyStopping callback that
#watches val_loss and uses restore_best_weights=True
def fit(model, training_dataset, validation_data, epochs, class_weight, learning_rate, patience=4):
    #epsilon of 1e-7 is the Keras Adam default, PyTorch's own default is 1e-8
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-7,
    )
    history = []
    best_val_loss = float("inf")
    best_weights = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        metrics = Metrics()
        for images, labels in training_dataset:
            optimizer.zero_grad()
            probabilities = model(images)
            losses = binary_crossentropy(probabilities, labels)
            #Each image is multiplied by the weight of its class, 0-normal and 1-sickle
            weights = labels * class_weight[1] + (1 - labels) * class_weight[0]
            weighted_losses = losses * weights
            weighted_losses.mean().backward()
            optimizer.step()
            metrics.update(weighted_losses.detach(), probabilities.detach(), labels)

        logs = metrics.results()
        logs.update({f"val_{k}": v for k, v in evaluate(model, validation_data).items()})
        history.append(logs)
        print(f"Epoch {epoch + 1}/{epochs} - " + " - ".join(f"{k}: {v:.4f}" for k, v in logs.items()))

        if best_weights is None:
            best_weights = copy.deepcopy(model.state_dict())

        wait += 1
        if logs["val_loss"] < best_val_loss:
            best_val_loss = logs["val_loss"]
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
        elif wait >= patience and epoch > 0:
            print(f"Epoch {epoch + 1}: early stopping")
            break

    model.load_state_dict(best_weights) #end on the full pass with the best validation performance
    return history


def plot_loss(history, *filenames):
    #Draws the training and validation loss of every epoch, with a dashed line on the best epoch, and saves it to each file given
    import matplotlib
    matplotlib.use("Agg") #draw straight to a file, no window needed
    import matplotlib.pyplot as plt

    loss = [logs["loss"] for logs in history]
    val_loss = [logs["val_loss"] for logs in history]
    epochs = range(1, len(loss) + 1)
    best = val_loss.index(min(val_loss)) + 1

    plt.figure()
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.axvline(best, linestyle="--", label=f"Best epoch ({best})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (binary cross-entropy)")
    plt.legend()
    for filename in filenames:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
