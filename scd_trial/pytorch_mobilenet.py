import os #to build the path of the downloaded weights file

import h5py #module to read the .h5 weight files that Keras uses
import torch
from torch import nn
from torch.nn import functional as F

#The same ImageNet weights that Keras downloads when weights="imagenet" is used
WEIGHTS_URL = (
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/"
    "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{alpha}_224_no_top.h5"
)


#Used to keep the number of filters divisible by 8, the same rule as the Keras MobileNetV2
def make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


#Keras' padding="same" with a stride of 2 adds 1 pixel on the bottom and right only.
#PyTorch can only pad both sides equally on its own, so the extra padding is added by hand.
class Conv2dSame(nn.Conv2d):
    def forward(self, x):
        return super().forward(F.pad(x, (0, 1, 0, 1)))


#Stands in for the Keras Add layer that joins the start of a block to its end
class Add(nn.Module):
    def forward(self, block_input, x):
        return block_input + x


#Batch normalisation with the same settings as the Keras MobileNetV2
def batch_norm(channels):
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.001)


class MobileNetV2(nn.Module):
    #Each row: output filters, stride, expansion, block id
    BLOCKS = [
        (16, 1, 1, 0),
        (24, 2, 6, 1), (24, 1, 6, 2),
        (32, 2, 6, 3), (32, 1, 6, 4), (32, 1, 6, 5),
        (64, 2, 6, 6), (64, 1, 6, 7), (64, 1, 6, 8), (64, 1, 6, 9),
        (96, 1, 6, 10), (96, 1, 6, 11), (96, 1, 6, 12),
        (160, 2, 6, 13), (160, 1, 6, 14), (160, 1, 6, 15),
        (320, 1, 6, 16),
    ]

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        #Every layer is stored by its Keras name and in the Keras order.
        #This keeps "the last 30 layers" pointing at exactly the same layers as before.
        self.named_layers = nn.ModuleDict()

        self.named_layers["input_layer"] = nn.Identity()
        channels = make_divisible(32 * alpha)
        self.named_layers["Conv1"] = Conv2dSame(3, channels, 3, stride=2, bias=False)
        self.named_layers["bn_Conv1"] = batch_norm(channels)
        self.named_layers["Conv1_relu"] = nn.ReLU6()

        for filters, stride, expansion, block_id in self.BLOCKS:
            channels = self._add_block(channels, filters, stride, expansion, block_id)

        last_channels = 1280 #stays at 1280 for every alpha of 1.0 and below
        self.named_layers["Conv_1"] = nn.Conv2d(channels, last_channels, 1, bias=False)
        self.named_layers["Conv_1_bn"] = batch_norm(last_channels)
        self.named_layers["out_relu"] = nn.ReLU6()

    def _add_block(self, in_channels, filters, stride, expansion, block_id):
        out_channels = make_divisible(int(filters * self.alpha))
        hidden = in_channels * expansion
        prefix = f"block_{block_id}_" if block_id else "expanded_conv_"

        if block_id: #the first block has no expand step
            self.named_layers[prefix + "expand"] = nn.Conv2d(in_channels, hidden, 1, bias=False)
            self.named_layers[prefix + "expand_BN"] = batch_norm(hidden)
            self.named_layers[prefix + "expand_relu"] = nn.ReLU6()

        if stride == 2:
            self.named_layers[prefix + "pad"] = nn.ZeroPad2d((0, 1, 0, 1))
            depthwise = nn.Conv2d(hidden, hidden, 3, stride=2, groups=hidden, bias=False)
        else:
            depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)

        self.named_layers[prefix + "depthwise"] = depthwise
        self.named_layers[prefix + "depthwise_BN"] = batch_norm(hidden)
        self.named_layers[prefix + "depthwise_relu"] = nn.ReLU6()
        self.named_layers[prefix + "project"] = nn.Conv2d(hidden, out_channels, 1, bias=False)
        self.named_layers[prefix + "project_BN"] = batch_norm(out_channels)

        if in_channels == out_channels and stride == 1:
            self.named_layers[prefix + "add"] = Add()
        return out_channels

    #The list of layers in order, like base_model.layers in Keras
    @property
    def layers(self):
        return list(self.named_layers.values())

    def forward(self, x):
        block_input = x
        for name, layer in self.named_layers.items():
            if name.endswith("_expand"):
                block_input = x #remembered so that the Add layer can join it back in
            if isinstance(layer, Add):
                x = layer(block_input, x)
            else:
                x = layer(x)
        return x


#Keras stores convolution weights as (height, width, in, out), PyTorch as (out, in, height, width)
def load_conv_kernel(layer, kernel):
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(kernel.transpose(3, 2, 0, 1)))


#Depthwise weights are (height, width, in, multiplier) in Keras, (in, multiplier, height, width) in PyTorch
def load_depthwise_kernel(layer, kernel):
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(kernel.transpose(2, 3, 0, 1)))


def is_depthwise(layer):
    return isinstance(layer, nn.Conv2d) and layer.groups > 1


def load_batch_norm(layer, gamma, beta, moving_mean, moving_variance):
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(gamma))
        layer.bias.copy_(torch.from_numpy(beta))
        layer.running_mean.copy_(torch.from_numpy(moving_mean))
        layer.running_var.copy_(torch.from_numpy(moving_variance))


#Loads the Keras ImageNet weights (the same starting point as weights="imagenet")
def load_imagenet_weights(base_model):
    alpha_text = "1.0" if base_model.alpha == 1.0 else f"{base_model.alpha:.2f}"
    url = WEIGHTS_URL.format(alpha=alpha_text)
    folder = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, os.path.basename(url))
    if not os.path.exists(path):
        torch.hub.download_url_to_file(url, path)

    with h5py.File(path, "r") as weights:
        #This file uses older layer names, so just like Keras, the layers are matched by their order
        layers_with_weights = [
            layer for layer in base_model.layers if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d))
        ]
        names_with_weights = [
            name for name in weights.attrs["layer_names"] if len(weights[name].attrs["weight_names"]) > 0
        ]

        for layer, name in zip(layers_with_weights, names_with_weights, strict=True):
            group = weights[name][name] #the file keeps each layer's weights in a group inside a group
            if isinstance(layer, nn.BatchNorm2d):
                load_batch_norm(
                    layer,
                    group["gamma:0"][()],
                    group["beta:0"][()],
                    group["moving_mean:0"][()],
                    group["moving_variance:0"][()],
                )
            elif is_depthwise(layer):
                load_depthwise_kernel(layer, group["depthwise_kernel:0"][()])
            else:
                load_conv_kernel(layer, group["kernel:0"][()])


#Loads the base model weights out of the .keras file saved by the original project.
#Keras numbers the layers of each type in order: conv2d, conv2d_1, conv2d_2 and so on.
def load_keras_base_weights(base_model, group):
    counts = {"conv2d": 0, "depthwise_conv2d": 0, "batch_normalization": 0}

    def next_group(kind):
        index = counts[kind]
        counts[kind] += 1
        return group[kind if index == 0 else f"{kind}_{index}"]["vars"]

    for layer in base_model.layers:
        if isinstance(layer, nn.BatchNorm2d):
            variables = next_group("batch_normalization")
            load_batch_norm(layer, *[variables[str(i)][()] for i in range(4)])
        elif is_depthwise(layer):
            load_depthwise_kernel(layer, next_group("depthwise_conv2d")["0"][()])
        elif isinstance(layer, nn.Conv2d):
            load_conv_kernel(layer, next_group("conv2d")["0"][()])
