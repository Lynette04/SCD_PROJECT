<<<<<<< HEAD
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models #layers to build the thinking parts of CNN, models to assemble full model of layers
from tensorflow.keras.applications import MobileNetV2 #import the pre trained model built into Keras
=======
from pytorch_training import build_pretrained_model, evaluate, fit, save_model, summary #the model and training tools shared by the scripts
>>>>>>> 6513c5efaf825d95b0dbf7206d8611868b7cc452
from step2_data_pipeline import get_datasets, get_class_weights #borrow the functions of the previous file

img_size = (224,224) #required size for images expected by MobileNetV2

def build_model(): #Function to build the model
    #MobileNetV2 is loaded with its pretrained ImageNet knowledge and without its final decision layer.
    #The foundational model is frozen, it should not be tampered with, just going to add necessary layers to it.
    #
    #The layers added to it are found in SickleCellClassifier in pytorch_training.py:
    #Normally, MobileNetV2's output for an image is a grid of numbers
    #The decision layer being built needs a compact list of numbers to work with
    #That's the purpose of the global average pooling. It compresses the grid to just a compact list of numbers per image.
    #
    #Dropout is used to randomly switch off some values during the training process
    #This is necessary to prevent overfitting of the model
    #It will switch of 30% of the values randomly
    #
    #The dense layer is also known as the fully connected layer. It uses each input to calculate a single output.
    #sigmoid: all output numbers should be between 0 and 1, without sigmoid the output could be a -5000 or 0.01 or anything.
    #The output represents how confident the model is in predicting an image whether its normal or sickle.
    model = build_pretrained_model("mobilenet_v2")

    print("Base model loaded. Number of layers:", len(model.base_model.layers))
    print("Full model Summary: \n")
    summary(model) #prints a table showing layers in the model and the number of parameters they have.

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

    #Actual training of the model using the training dataset to learn, the validation set for comparison
    #it uses the class weights here
    #Adam is the algorithm responsible for adjusting the trainable parameters based on mistakes, it is very reliable in DL.
    #learning rate of 0.001 to control the adjustment size.
    #binary crossentropy is a mathematical formula designed to work well with outputs that are probabilities between 0 and 1. It is used specifically for 2 class problems(Normal vs Sickle)
    #Early stopping checks on the validation loss(how wrong the model is in recognising sickle cells),
    #patience = 4, if it makes 4 full passes through all the training data and has no improvement, it should stop early.
    #whatever full pass had the best validation performance, it should end on it.
    #epoch is a full pass through all training data
    history = fit(
        model,
        training_dataset,
        validation_data= validation_dataset,
        epochs=15,
        class_weight=class_weights,
        learning_rate=1e-3,
        patience=4
    )
<<<<<<< HEAD
    
    

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
    plt.savefig("training_loss.png")
        
=======


>>>>>>> 6513c5efaf825d95b0dbf7206d8611868b7cc452
    print("\n -----Final evaluation on Test Set -----")
    #after training, feed the model with test dataset to get its honest performance
    test_results = evaluate(model, test_dataset)
    for metric_name, value in test_results.items():
        print(f"{metric_name}: {value:.4f}")

    #save the entire trained model to a single file to prevent retraining whenever it is required for use.
    save_model(model, "sickle_cell_model.pt")
    print("\n Model is saved to sickle_cell_model.pt")
