import tensorflow as tf
from tensorflow.keras import layers
from step2_data_pipeline import get_datasets, get_class_weights

model = tf.keras.models.load_model("sickle_cell_model.keras") #load the saved model

model.summary()

base_model = model.layers[1] #[1] represents the base model index in our trained model. [0] represents the input layer.
print(f"Base model has {len(base_model.layers)} layers.")

base_model.trainable = True #temporarily unfreeze the entire base model.

unfreezed_layers = len(base_model.layers)-30

for i, layer in enumerate(base_model.layers):
    if i < unfreezed_layers:
        layer.trainable = False
    else:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False #Because we have a small dataset, training involving the batchnormalisation layers can become very unstable
        else:
            layer.trainable = True
            
trainable_layer_count =sum(1 for layer in base_model.layers if layer.trainable)
print(f"Layers actually unfrozen: {trainable_layer_count} out of {len(base_model.layers)}")            

#Because we have changed the number of trainable layers, the model has to be recompiled but with a smaller learning rate
model.compile(
    #Here we use the Adam optimizer at a learning rate of 0.00001 so that small corrections are made each time the model is wrong.
    #It is smaller than the previous one of 0.001 because we are adjusting numbers in the base model which contain knowledge from 1.4million images. 
    #So a smaller learning rate is needed to prevent spoiling the good knowledge of the base model.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5), 
    
    #To calculate how wrong each guess was, we use the formula of binary_crossentropy because we are dealing with two possible answers either sickle or normal with a probability between 0 and 1.
    loss = "binary_crossentropy",
    metrics=["accuracy",
              tf.keras.metrics.Precision(name="precision"),
              tf.keras.metrics.Recall(name="recall")
    ]
)

total_parameters = model.count_params()
trainable_parameters = sum(tf.size(w).numpy() for w in model.trainable_weights)
print(f"Total parameters: {total_parameters:,} | Trainable parameters now: {trainable_parameters:,}")

#Training the model: fine-tuned version
training_dataset, validation_dataset, test_dataset, class_names = get_datasets()
class_weights, class_counts = get_class_weights()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)
print("Let's start the fine tuning..\n")
history = model.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs=12,
    class_weight = class_weights,
    callbacks = [early_stop]
)

#Using the test images to evaluate the fine-tuned model
test_results = model.evaluate(test_dataset, return_dict=True)
for metric_name, value in test_results.items():
    print(f"{metric_name}:{value:.4f}")

model.save("Sickle_cell_fine_tuned-mode.keras")
print("The model is saved in the desktop")

