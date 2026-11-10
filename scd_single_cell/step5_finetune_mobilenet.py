from pytorch_training import (
        count_params,
        count_trainable_params,
        evaluate,
        fit,
        load_model,
        plot_loss,
        save_model,
        set_random_seed,
        unfreeze_last_layers,
    )
from step1_data_pipeline import get_datasets, get_class_weights

set_random_seed(42)

# MobileNetV2 normalizes pixel values [0, 255] to [-1.0, 1.0]
def mobilenet_preprocess(images):
    return images / 127.5 - 1.0

# Unfreeze the last 8 layers of the MobileNetV2 backbone
UNFREEZE_LAST = 8

if __name__ == "__main__":
    print("Loading baseline model: scd_singlecell_mobilenet_v2.pt...")
    model = load_model("scd_singlecell_mobilenet_v2.pt")
    model.dropout.p = 0.4  # Slightly higher dropout to prevent overfitting

    base_model = model.base_model
    trainable_count = unfreeze_last_layers(model, UNFREEZE_LAST)
    print(f"Base model total layers: {len(base_model.layers)}")
    print(f"Layers actually unfrozen: {trainable_count}")
    print(f"Total params: {count_params(model):,} | Trainable params: {count_trainable_params(model):,}")

    print("\nLoading single-cell dataset (MobileNetV2 preprocess)...")
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets(preprocess_fn=mobilenet_preprocess)
    class_weights, class_counts = get_class_weights()

    print("\n--- Starting Fine-Tuning: MobileNetV2 ---")
    history = fit(
        model,
        training_dataset,
        validation_data=validation_dataset,
        epochs=10,
        class_weight=class_weights,
        learning_rate=1e-5,  # 100x smaller learning rate to preserve features
        weight_decay=1e-4,
        patience=4,
    )

    plot_loss(history, "singlecell_mobilenet_ft_loss.png")
    print("\nLoss curve saved to singlecell_mobilenet_ft_loss.png")

    print("\n----- Final Evaluation on Test Cells -----")
    test_results = evaluate(model, test_dataset)
    for metric_name, value in test_results.items():
        print(f"{metric_name}: {value:.4f}")

    save_model(model, "scd_singlecell_mobilenet_v2_finetuned.pt")
    print("\nFine-tuned model saved to scd_singlecell_mobilenet_v2_finetuned.pt")