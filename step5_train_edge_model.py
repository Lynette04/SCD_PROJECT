from pytorch_training import build_pretrained_model, evaluate, fit, plot_loss, save_model, summary
from step2_data_pipeline import get_datasets, get_class_weights

img_size= (224,224)

def build_edge_model():
    #Alpha controls how many filters exist at each layer. For the edge model, we need a smaller alpha value so that it is much lighter and faster but accuracy might decrease
    #The added layers are the same as in step 3, with the same 30% dropout
    return build_pretrained_model("mobilenet_v2", alpha=0.35)

if __name__== "__main__":
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets()
    class_weights, class_counts = get_class_weights()

    model = build_edge_model()
    summary(model)

    print("Let's train the edge model....")
    history = fit(
        model,
        training_dataset,
        validation_data= validation_dataset,
        epochs=15,
        class_weight=class_weights,
        learning_rate=1e-3,
        patience=4
    )
    plot_loss(history, "training_loss_edge.png")

    print("FINAL EVALUATION USING THE TEST SET")
    test_results = evaluate(model, test_dataset)
    for metric_name, value in test_results.items():
        print(f"{metric_name} | {value:.4f}")

    save_model(model, "sickle_cell_edge_model.pt")
    print("The edge model has been saved as sickle_cell_edge_model.pt ")
