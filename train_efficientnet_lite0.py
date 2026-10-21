import tensorflow as tf
from tensorflow.keras import layers, models  #type: ignore
import keras_hub
from step2_data_pipeline import get_datasets, get_class_weights, efficientnet_preprocess

img_size = (224,224)
tf.keras.utils.set_random_seed(42)

def build_model():
    base_model = keras_hub.models.Backbone.from_preset("efficientnet_lite0_ra_imagenet")
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=img_size+ (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs,outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss = "binary_crossentropy",
        metrics = ["accuracy",
                   tf.keras.metrics.Precision(name="precision"),
                   tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()
    return model

if __name__ == "__main__":
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets(efficientnet_preprocess)
    class_weights, class_counts = get_class_weights()
    
    print("Class names are: ", class_names)
    print("Class weights are: ",class_weights)
    
    model = build_model()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )
    
    history = model.fit(
        training_dataset,
        validation_data = validation_dataset,
        epochs=15,
        class_weight=class_weights,
        callbacks=[early_stop]
    )
    
    print("\n -----Final evaluation on Test Set-----")
    test_results = model.evaluate(test_dataset, return_dict=True)
    for metric_name, value in test_results.items():
        print(f"{metric_name}: {value:.4f}")
    
    model.save("sickle_cell_effnet_lite0.keras")
    print("\n Model is saved to sickle_cell_effnet_lite0.keras")
        