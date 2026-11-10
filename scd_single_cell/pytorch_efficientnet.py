import timm #library that holds the pretrained EfficientNet-Lite0
import torch
from torch import nn

from pytorch_mobilenet import Add


#Batch normalisation with the same settings as the keras_hub EfficientNet-Lite0
def batch_norm(channels):
    return nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)


class EfficientNetLite0(nn.Module):
    #Each row: repeats, kernel size, input filters, output filters, expansion, stride
    STACKS = [
        (1, 3, 32, 16, 1, 1),
        (2, 3, 16, 24, 6, 2),
        (2, 5, 24, 40, 6, 2),
        (3, 3, 40, 80, 6, 2),
        (3, 5, 80, 112, 6, 1),
        (4, 5, 112, 192, 6, 2),
        (1, 3, 192, 320, 6, 1),
    ]

    def __init__(self):
        super().__init__()
        #Every layer is stored by its keras_hub name and in the same order,
        #so that "the last 30 layers" points at exactly the same layers as before.
        self.named_layers = nn.ModuleDict()

        self.named_layers["input_layer"] = nn.Identity()
        self.named_layers["stem_conv_pad"] = nn.ZeroPad2d(1)
        self.named_layers["stem_conv"] = nn.Conv2d(3, 32, 3, stride=2, bias=False)
        self.named_layers["stem_bn"] = batch_norm(32)
        self.named_layers["stem_activation"] = nn.ReLU6()

        for stack, (repeats, kernel, in_channels, out_channels, expansion, stride) in enumerate(self.STACKS, start=1):
            for repeat in range(repeats):
                prefix = f"block{stack}{'abcd'[repeat]}_"
                block_in = in_channels if repeat == 0 else out_channels
                block_stride = stride if repeat == 0 else 1
                self._add_block(prefix, block_in, out_channels, kernel, expansion, block_stride)

        self.named_layers["top_conv"] = nn.Conv2d(320, 1280, 1, bias=False)
        self.named_layers["top_bn"] = batch_norm(1280)
        self.named_layers["top_activation"] = nn.ReLU6()

    def _add_block(self, prefix, in_channels, out_channels, kernel, expansion, stride):
        hidden = in_channels * expansion

        if expansion > 1: #the first block has no expand step
            self.named_layers[prefix + "expand_conv"] = nn.Conv2d(in_channels, hidden, 1, bias=False)
            self.named_layers[prefix + "expand_bn"] = batch_norm(hidden)
            self.named_layers[prefix + "expand_activation"] = nn.ReLU6()

        self.named_layers[prefix + "dwconv_pad"] = nn.ZeroPad2d(kernel // 2)
        self.named_layers[prefix + "dwconv"] = nn.Conv2d(
            hidden, hidden, kernel, stride=stride, groups=hidden, bias=False
        )
        self.named_layers[prefix + "dwconv_bn"] = batch_norm(hidden)
        self.named_layers[prefix + "dwconv_activation"] = nn.ReLU6()
        self.named_layers[prefix + "project"] = nn.Conv2d(hidden, out_channels, 1, bias=False)
        self.named_layers[prefix + "project_bn"] = batch_norm(out_channels)

        if in_channels == out_channels and stride == 1:
            self.named_layers[prefix + "add"] = Add()

    #The list of layers in order, like base_model.layers in Keras
    @property
    def layers(self):
        return list(self.named_layers.values())

    def forward(self, x):
        block_input = x
        for name, layer in self.named_layers.items():
            if name.endswith("_expand_conv"):
                block_input = x #remembered so that the Add layer can join it back in
            if isinstance(layer, Add):
                x = layer(block_input, x)
            else:
                x = layer(x)
        return x


#Copies the ImageNet weights from timm's EfficientNet-Lite0 into the layers above.
#timm's copy and the keras_hub preset are the same "ra" weights in a different layout.
def load_imagenet_weights(base_model):
    source = timm.create_model("efficientnet_lite0.ra_in1k", pretrained=True)
    layers = base_model.named_layers

    def copy(target_name, source_layer):
        target = layers[target_name]
        with torch.no_grad():
            target.weight.copy_(source_layer.weight)
            if isinstance(target, nn.BatchNorm2d):
                target.bias.copy_(source_layer.bias)
                target.running_mean.copy_(source_layer.running_mean)
                target.running_var.copy_(source_layer.running_var)

    copy("stem_conv", source.conv_stem)
    copy("stem_bn", source.bn1)

    for stack, stack_blocks in enumerate(source.blocks, start=1):
        for repeat, block in enumerate(stack_blocks):
            prefix = f"block{stack}{'abcd'[repeat]}_"
            if stack == 1: #timm calls this block by different names since it has no expand step
                copy(prefix + "dwconv", block.conv_dw)
                copy(prefix + "dwconv_bn", block.bn1)
                copy(prefix + "project", block.conv_pw)
                copy(prefix + "project_bn", block.bn2)
            else:
                copy(prefix + "expand_conv", block.conv_pw)
                copy(prefix + "expand_bn", block.bn1)
                copy(prefix + "dwconv", block.conv_dw)
                copy(prefix + "dwconv_bn", block.bn2)
                copy(prefix + "project", block.conv_pwl)
                copy(prefix + "project_bn", block.bn3)

    copy("top_conv", source.conv_head)
    copy("top_bn", source.bn2)
