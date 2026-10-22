import tensorflow as tf
import keras_hub
from tensorflow.keras import layers #type: ignore
from step2_data_pipeline import get_datasets, get_class_weights,efficientnet_preprocess

tf.keras.utils.set_random_seed(42)

UNFREEZE_LAST = 30

model = tf.keras.models.load_model("sickle_cell_effnet_lite0.keras")

base_model = model.layers[1]
print(f"Base model has {len(base_model.layers)} layers.")

base_model.trainable = True

first_unfrozen = len(base_model.layers) - UNFREEZE_LAST

for i, lr in enumerate(base_model.layers):
    if i < first_unfrozen:
        lr.trainable = False
    elif isinstance(lr, layers.BatchNormalization):
        lr.trainable = False
    else:
        lr.trainable = True
        
trainable_layer_count = sum(1 for lr in base_model.layers if lr.trainable)
print(f"Layers actually unfrozen: {trainable_layer_count} out of {len(base_model.layers)}")

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy",
               tf.keras.metrics.Precision(name="precision"),
               tf.keras.metrics.Recall(name="recall")                 
                                ]
)

total_parameters = model.count_params()
trainable_parameters = sum(tf.size(w).numpy() for w in model.trainable_weights)
print(f"Total parameters: {total_parameters:,} | Trainable parameters now: {trainable_parameters:,}")

training_dataset, validation_dataset, test_dataset, class_names = get_datasets(efficientnet_preprocess)
class_weights, class_counts = get_class_weights()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights= True
)

print("Let's start the fine tuning..\n")
history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=12,
    class_weight=class_weights,
    callbacks=[early_stop]
)

test_results = model.evaluate(test_dataset, return_dict=True)
for metric_name, value in test_results.items():
    print(f"{metric_name}: {value:.4f}")

model.save(f"sickle_celll_effnet_lite0_ft{UNFREEZE_LAST}.keras")
print(f"Model saved to sickle_cell_effnet_lite0_ft{UNFREEZE_LAST}.keras")