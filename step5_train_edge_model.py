import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from step2_data_pipeline import get_datasets, get_class_weights

img_size= (224,224)

def build_edge_model():
    base_model = MobileNetV2(
        input_shape = img_size + (3,),
        alpha = 0.35, #Alpha controls how many filters exist at each layer. For the edge model, we need a smaller alpha value so that it is much lighter and faster but accuracy might decrease
        include_top = False,
        weights= "imagenet"
    )
    
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=img_size + (3,))
    x= base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) #same 30% of training numbers 
    outputs = layers.Dense(1,activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss= "binary_crossentropy",
        metrics = ["accuracy",
                   tf.keras.metrics.Precision(name="precision"),
                   tf.keras.metrics.Recall(name="recall")  
        ]
    )
    
    return model

if __name__== "__main__":
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets()
    class_weights, class_counts = get_class_weights()
    
    model = build_edge_model()
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor= "val_loss",
        patience=4,
        restore_best_weights=True
    )
    
    print("Let's train the edge model....")
    history = model.fit(
        training_dataset,
        validation_data= validation_dataset,
        epochs=15,
        class_weight=class_weights,
        callbacks=[early_stop]
    )
    
    print("FINAL EVALUATION USING THE TEST SET")
    test_results = model.evaluate(test_dataset, return_dict=True)
    for metric_name, value in test_results.items():
        print(f"{metric_name} | {value:.4f}")
    
    model.save("sickle_cell_edge_model.keras")
    print("The edge model has been saved as sickle_cell_edge_model.keras ")    