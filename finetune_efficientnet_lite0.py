from pytorch_training import count_params, count_trainable_params, evaluate, fit, load_model, save_model, set_random_seed, unfreeze_last_layers
from step2_data_pipeline import get_datasets, get_class_weights,efficientnet_preprocess #data pipeline

#random seed = 42, so that a re-run gives the same result.
set_random_seed(42)

#layers to be activated (30 last layers)
UNFREEZE_LAST = 30

#load the frozen layer efficientnet model
model = load_model("sickle_cell_effnet_lite0.pt")

#the backbone inside the model
base_model = model.base_model
print(f"Base model has {len(base_model.layers)} layers.")

#early layers keep their ImageNet knowledge intact, Batchnormalization layers also stay frozen
#and the later layers are allowed to adjust to blood cell images
trainable_layer_count = unfreeze_last_layers(model, UNFREEZE_LAST)

#print number of unfrozen layers
print(f"Layers actually unfrozen: {trainable_layer_count} out of {len(base_model.layers)}")

#print the total parameters and trainable parameters
total_parameters = count_params(model)
trainable_parameters = count_trainable_params(model)
print(f"Total parameters: {total_parameters:,} | Trainable parameters now: {trainable_parameters:,}")

#load data with the EfficientNet preprocessing
training_dataset, validation_dataset, test_dataset, class_names = get_datasets(efficientnet_preprocess)
class_weights, class_counts = get_class_weights()

#stop early if validation loss has not improved for 4 passes and keep best weights
print("Let's start the fine tuning..\n")
history = fit(
    model,
    training_dataset,
    validation_data=validation_dataset,
    epochs=12,
    class_weight=class_weights,
    learning_rate=1e-5, #same learning rate as MobileNetV2 fine tuned model
    patience=4
)

#evaluate on the test images
test_results = evaluate(model, test_dataset)
for metric_name, value in test_results.items():
    print(f"{metric_name}: {value:.4f}")

#Save the model file
save_model(model, f"sickle_cell_effnet_lite0_ft{UNFREEZE_LAST}.pt")
print(f"Model saved to sickle_cell_effnet_lite0_ft{UNFREEZE_LAST}.pt")
