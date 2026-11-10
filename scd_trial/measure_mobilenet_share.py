from pytorch_training import count_params, count_trainable_params, load_model, unfreeze_last_layers

#loading the frozen MobileNetV2 model, no training will happen here
model = load_model("sickle_cell_model.pt")

#The MobileNetV2 backbone
base_model = model.base_model

#Using only last 30 layers like in the fine tuning script for the second model variant of mobilenetv2
unfreeze_last_layers(model, 30)

#Sum up the weights that would train inside the backbone, and compare with the backbone's total

trainable = count_trainable_params(base_model)
total = count_params(base_model)
print(f"Backbone trainable weights:{trainable} of {total} = {round(100 * trainable / total, 1)} percent")
