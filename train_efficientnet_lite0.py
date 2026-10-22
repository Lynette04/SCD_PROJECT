import tensorflow as tf
from tensorflow.keras import layers, models  #type: ignore
import keras_hub #library that holds the pretrained EfficientNet-Lite0
from step2_data_pipeline import get_datasets, get_class_weights, efficientnet_preprocess

img_size = (224,224) #resizing all images to 224 by 224 pixels
tf.keras.utils.set_random_seed(42)

def build_model():
    #download the EfficientNet-Lite0 with its learned ImgeNet knowledge 
    base_model = keras_hub.models.Backbone.from_preset("efficientnet_lite0_ra_imagenet")
    base_model.trainable = False #freeze so that learned knowledge isn't touched
    
    inputs = tf.keras.Input(shape=img_size+ (3,)) #entry point for images, 224 by 224 pixels, 3 colour channels
    x = base_model(inputs, training=False) #pass the input through EfficientNet
    x = layers.GlobalAveragePooling2D()(x) #squeeze each image grid into a compact list of 1280 numbers
    x = layers.Dropout(0.3)(x) #randomly switch off 30% of the values during training to reduce overfitting
    outputs = layers.Dense(1, activation="sigmoid")(x) #final decision: one number between 0 and 1 (0 is normal and 1 is sickle)
    model = models.Model(inputs,outputs) #join the inputs and the ouputs
    
    #set up how the model will learn
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),# #Here we use the Adam optimizer at a learning rate of 0.00001 so that small corrections are made each time the model is wrong.
            #It is smaller than the previous one of 0.001 because we are adjusting numbers in the base model which contain knowledge from 1.4million images. 
            #So a smaller learning rate is needed to prevent spoiling the good knowledge of the base model.
        loss = "binary_crossentropy", #measures how wrong each guess is for a two-class problem.
        metrics = ["accuracy",
                   tf.keras.metrics.Precision(name="precision"), #of the images called sickle, how many are actually sickle
                   tf.keras.metrics.Recall(name="recall")], #of the sickle images, how many were found
    )
    model.summary() #print a table of the layers and parameter counts
    return model 

if __name__ == "__main__":
    #load the 3 image sets  with the EfficientNet preprocessing
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets(efficientnet_preprocess)
    class_weights, class_counts = get_class_weights() #give the smaller normal class a bigger weight
    
    print("Class names are: ", class_names)
    print("Class weights are: ",class_weights)
    
    model = build_model()
    
    #stop training when there are no more improvements
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )
    #training now
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
        