from pytorch_training import build_pretrained_model, evaluate, fit, plot_loss, save_model, set_random_seed, summary #the model and training tools shared by the scripts
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
    
    
    #define hyperparameter grid to explore
    learning_rates = [1e-4, 1e-3, 3e-3]
    dropouts = [0.2, 0.4]
    
    best_val_loss = float("inf")
    best_config = None
    best_model = None
    best_history = None
    
    print("\n--- STARTING HYPERPARAMETER TUNING---")
    for lr in learning_rates:
        for drop in dropouts:
            print(f"\nTesting: learning_rate={lr}, dropout={drop}")
            
            #Reset seed so every trial gets the same starting conditions
            set_random_seed(42)
            
            #build model with trial's dropout
            model = build_pretrained_model("mobilenet_v2", dropout_rate=drop)
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
                learning_rate=lr,
                patience=4,
                verbose=False
            )
            
            min_val_loss = min(h["val_loss"] for h in history)
            print(f"Result -> Best val_loss: {min_val_loss:.4f}")
            
            if min_val_loss < best_val_loss:
                best_val_loss = min_val_loss
                best_config = {"learning_rate": lr, "dropout": drop}
                best_model = model
                best_history = history
    if best_history is not None:            
        plot_loss(history, "loss_curves.png", "training_loss.png") #save a picture of how the loss changed over the epochs

    print("\n -----Final evaluation on Test Set -----")
    #after training, feed the model with test dataset to get its honest performance
    test_results = evaluate(model, test_dataset)
    for metric_name, value in test_results.items():
        print(f"{metric_name}: {value:.4f}")

    #save the entire trained model to a single file to prevent retraining whenever it is required for use.
    save_model(best_model, "sickle_cell_model.pt")
    print("\n Model is saved to sickle_cell_model.pt")
