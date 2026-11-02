from pytorch_training import count_params, count_trainable_params, evaluate, fit, load_model, save_model, summary, unfreeze_last_layers
from step2_data_pipeline import get_datasets, get_class_weights

model = load_model("sickle_cell_model.pt") #load the saved model

summary(model)

base_model = model.base_model #the base model inside our trained model
print(f"Base model has {len(base_model.layers)} layers.")

#Unfreeze the last 30 layers of the base model.
#The batchnormalisation layers stay frozen. Because we have a small dataset, training involving the batchnormalisation layers can become very unstable
trainable_layer_count = unfreeze_last_layers(model, 30)
print(f"Layers actually unfrozen: {trainable_layer_count} out of {len(base_model.layers)}")

total_parameters = count_params(model)
trainable_parameters = count_trainable_params(model)
print(f"Total parameters: {total_parameters:,} | Trainable parameters now: {trainable_parameters:,}")

#Training the model: fine-tuned version
training_dataset, validation_dataset, test_dataset, class_names = get_datasets()
class_weights, class_counts = get_class_weights()

print("Let's start the fine tuning..\n")
history = fit(
    model,
    training_dataset,
    validation_data = validation_dataset,
    epochs=12,
    class_weight = class_weights,
    #Here we use the Adam optimizer at a learning rate of 0.00001 so that small corrections are made each time the model is wrong.
    #It is smaller than the previous one of 0.001 because we are adjusting numbers in the base model which contain knowledge from 1.4million images.
    #So a smaller learning rate is needed to prevent spoiling the good knowledge of the base model.
    learning_rate = 1e-5,
    patience = 4
)

#Using the test images to evaluate the fine-tuned model
test_results = evaluate(model, test_dataset)
for metric_name, value in test_results.items():
    print(f"{metric_name}:{value:.4f}")

save_model(model, "Sickle_cell_fine_tuned-mode.pt")
print("The model is saved in the desktop")
