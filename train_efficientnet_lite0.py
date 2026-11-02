from pytorch_training import build_pretrained_model, evaluate, fit, save_model, set_random_seed, summary
from step2_data_pipeline import get_datasets, get_class_weights, efficientnet_preprocess

img_size = (224,224) #resizing all images to 224 by 224 pixels
set_random_seed(42)

def build_model():
    #download the EfficientNet-Lite0 with its learned ImgeNet knowledge
    #The base model is frozen so that learned knowledge isn't touched.
    #The added layers are the same as MobileNetV2: squeeze each image grid into a compact list of 1280 numbers,
    #randomly switch off 30% of the values during training to reduce overfitting,
    #and make a final decision: one number between 0 and 1 (0 is normal and 1 is sickle)
    model = build_pretrained_model("efficientnet_lite0")
    summary(model) #print a table of the layers and parameter counts
    return model

if __name__ == "__main__":
    #load the 3 image sets  with the EfficientNet preprocessing
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets(efficientnet_preprocess)
    class_weights, class_counts = get_class_weights() #give the smaller normal class a bigger weight

    print("Class names are: ", class_names)
    print("Class weights are: ",class_weights)

    model = build_model()

    #training now, with Adam at a learning rate of 0.001 and binary crossentropy to measure how wrong each guess is for a two-class problem.
    #stop training when there are no more improvements
    history = fit(
        model,
        training_dataset,
        validation_data = validation_dataset,
        epochs=15,
        class_weight=class_weights,
        learning_rate=1e-3,
        patience=4
    )

    print("\n -----Final evaluation on Test Set-----")
    test_results = evaluate(model, test_dataset)
    for metric_name, value in test_results.items():
        print(f"{metric_name}: {value:.4f}")

    save_model(model, "sickle_cell_effnet_lite0.pt")
    print("\n Model is saved to sickle_cell_effnet_lite0.pt")
