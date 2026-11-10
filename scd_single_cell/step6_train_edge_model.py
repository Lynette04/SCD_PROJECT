from pytorch_training import (
        build_pretrained_model,
        count_params,
        count_trainable_params,
        evaluate,
        fit,
        plot_loss,
        save_model,
        set_random_seed,
        summary,
    )
from step1_data_pipeline import get_datasets, get_class_weights

set_random_seed(42)

# MobileNetV2 normalizes pixel values [0, 255] to [-1.0, 1.0]
def mobilenet_preprocess(images):
    return images / 127.5 - 1.0

def build_edge_model():
    # MobileNetV2 with alpha=0.35 (slashes channels by ~65% for Raspberry Pi deployment)
    model = build_pretrained_model("mobilenet_v2", alpha=0.35,
    dropout_rate=0.3)
    summary(model)
    return model

if __name__ == "__main__":
    print("Loading SKDrepaData-v2-clean datasets for Edge MobileNetV2 (alpha=0.35)...")
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets(preprocess_fn=mobilenet_preprocess)
    class_weights, class_counts = get_class_weights()

    model = build_edge_model()
    print(f"Total parameters: {count_params(model):,} | Trainable parameters: {count_trainable_params(model):,}")

    print("\n--- Starting Training on Edge Variant (alpha=0.35) ---")
    history = fit(
        model,
        training_dataset,
        validation_data=validation_dataset,
        epochs=12,
        class_weight=class_weights,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=4,
    )

    plot_loss(history, "singlecell_edge_loss.png")
    print("\nLoss curve saved to singlecell_edge_loss.png")

    print("\n----- Final Evaluation on Unseen Test Cells -----")
    test_results = evaluate(model, test_dataset)
    for metric_name, value in test_results.items():
        print(f"{metric_name}: {value:.4f}")

    save_model(model, "scd_singlecell_edge_mobilenet_035.pt")
    print("\nEdge model saved to scd_singlecell_edge_mobilenet_035.pt")