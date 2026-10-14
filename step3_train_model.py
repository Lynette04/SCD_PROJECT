import tensorflow as tf
from tensorflow.keras import layers, models #layers to build the thinking parts of CNN, models to assemble full model of layers
from tensorflow.keras.applications import MobileNetV2 #import the pre trained model built into Keras
from step2_data_pipeline import get_datasets, get_class_weights #borrow the functions of the previous file

img_size = (224,224) #required size for images expected by MobileNetV2

def build_model(): #Function to build the model
    base_model = MobileNetV2(
        input_shape = img_size + (3,), #tells MobileNetV2 the input shape to expect 224 by 224 , 3 color channels (RGB) shape: (224,224,3)
        include_top= False, #dont include the final decision layer of MobileNetV2
        weights="imagenet" #tell Keras to download the pretrained knowledge
    )
    
    base_model.trainable = False #the foundational model should not be tampered with, just going to add necessary layers to it 
    
    #Create a point of entry for a batch of images.
    #Its shape should be the same as the pipeline (224,224,3)
    inputs = tf.keras.Input(shape=img_size + (3,))
    
    #pass the input through the base_model but nothing happens
    x = base_model(inputs, training=False)
    
    #Normally, MobileNetV2's output for an image is a grid of numbers
    #The decision layer being built needs a compact list of numbers to work with
    #That's the purpose of GlobalAveragePooling2D. It compresses the grid to just a compact list of numbers per image.
    #It takes in the previous x and passes on its result to the next section
    x = layers.GlobalAveragePooling2D()(x)
    
    #Dropout is used to randomly switch off some values during the training process
    #This is necessary to prevent overfitting of the model
    #It will switch of 30% of the values randomly
    x = layers.Dropout(0.3)(x)
    
    #Create a variable called outputs to store the final prediction result
    #Dense layer is also known as the fully connected layer. It uses each input to calculate a single output.
    #1- a single number as output, activation=sigmoid: all output numbers should be between 0 and 1, without sigmoid the output could be a -5000 or 0.01 or anything, sigmoid makes all the output to be between 0 and 1.
    #(x) feeds data from the previous layer into dense layer
    #outputs represents how confident the model is in predicting an image whether its normal or sickle.  
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    #Create an object to assemble everything from the inputs to the outputs.
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3), #algorithm responsible for adjusting the trainable parameters based on mistakes, using the Adam optimiser which is very reliable in DL.
        #learning rate of 0.001 to control the adjustment size.
        loss="binary_crossentropy", #binary crossentropy is a mathematical formula designed to work well with outputs that are probabilities between 0 and 1. It is used specifically for 2 class problems(Normal vs Sickle)
        metrics=["accuracy", #metrics used to monitor the progress of the model while training e.g accuracy, precision and recall.
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ],
    )
    
    print("Base model loaded. Number of layers:", len(base_model.layers))
    print("Full model Summary: \n")
    model.summary() #built-in method that prints a table showing layers in the model, their output shapes and number of trainable parameters they have.
    
    return model

#main function to test that the build_model function works.
if __name__== "__main__":
    #load the datasets and class weights from data_pipeline file
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets()
    class_weights, class_counts= get_class_weights()
    
    print("Class names are: ", class_names)
    print("Class weights are: ", class_weights)
    
    #build the model using build_model function
    model = build_model()
    
    #Create a callback which checks on the validation loss(how wrong the model is in recognising sickle cells),
    #patience = 4, if it makes 4 full passes through all the training data and has no improvement, it should stop early.
    #restore_best_weights: whatever full pass had the best validation performance, it should end on it.
    #epoch is a full pass through all training data
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )
    
    #Actual training of the model using the training dataset to learn, the validation set for comparison
    #it uses the class weights and the callback here
    history = model.fit(
        training_dataset,
        validation_data= validation_dataset,
        epochs=15,
        class_weight=class_weights,
        callbacks=[early_stop]
    )
    
    
    print("\n -----Final evaluation on Test Set -----")
    #after training, feed the model with test dataset to get its honest performance
    test_results = model.evaluate(test_dataset, return_dict=True)
    for metric_name, value in test_results.items():
        print(f"{metric_name}: {value:.4f}")
    
    #save the entire trained model to a single file to prevent retraining whenever it is required for use.    
    model.save("sickle_cell_model.keras")
    print("\n Model is saved to sickle_cell_model.keras")    
