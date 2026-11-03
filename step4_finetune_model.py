<<<<<<< HEAD
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import layers
=======
from pytorch_training import count_params, count_trainable_params, evaluate, fit, load_model, save_model, summary, unfreeze_last_layers
>>>>>>> 6513c5efaf825d95b0dbf7206d8611868b7cc452
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
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
best = val_loss.index(min(val_loss)) + 1
plt.axvline(best, linestyle="--", label=f"Best epoch ({best})")
plt.xlabel("Epoch")
plt.ylabel("Loss (binary cross-entropy)")
plt.legend()
plt.savefig("loss_curves.png", dpi=200, bbox_inches="tight")
plt.savefig("training_loss_finetune.png")
    
#Using the test images to evaluate the fine-tuned model
test_results = evaluate(model, test_dataset)
for metric_name, value in test_results.items():
    print(f"{metric_name}:{value:.4f}")

save_model(model, "Sickle_cell_fine_tuned-mode.pt")
print("The model is saved in the desktop")
