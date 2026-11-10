from pytorch_training import build_pretrained_model, evaluate, plot_loss, fit, save_model, set_random_seed, summary
from step1_data_pipeline import get_datasets, get_class_weights 

set_random_seed(42)

# MobileNetV2 normalizes pixel values [0, 255] to [-1.0, 1.0]
def mobilenet_preprocess(images):
    return images / 127.5 - 1.0

def build_model():
    # MobileNetV2 backbone with frozen feature extractor
    model = build_pretrained_model("mobilenet_v2",
    dropout_rate=0.3)
    summary(model)
    return model

if __name__ == "__main__":
    print("Loading SKDrepaData-v2-clean datasets for MobileNetV2...")
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets(preprocess_fn=mobilenet_preprocess)
    class_weights, class_counts = get_class_weights()
    print("Class names are:", class_names)
    print("Class weights are:", class_weights)

    model = build_model()

    print("\n--- Starting Training on Single-Cell Dataset (MobileNetV2) ---")
    history = fit(
        model,
        training_dataset,
        validation_data=validation_dataset,
        epochs=10,
        class_weight=class_weights,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=3,
    )

    plot_loss(history, "singlecell_mobilenet_loss.png")
    print("\nLoss curve saved to singlecell_mobilenet_loss.png")

    print("\n----- Final Evaluation on Unseen Test Cells -----")
    test_results = evaluate(model, test_dataset)
    for metric_name, value in test_results.items():
        print(f"{metric_name}: {value:.4f}")

    save_model(model, "scd_singlecell_mobilenet_v2.pt")
    print("\nModel saved to scd_singlecell_mobilenet_v2.pt")