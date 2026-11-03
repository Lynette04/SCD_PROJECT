<<<<<<< HEAD
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2

=======
from pytorch_training import build_pretrained_model, evaluate, fit, save_model, summary
>>>>>>> 6513c5efaf825d95b0dbf7206d8611868b7cc452
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
    plt.savefig("training_loss_edge.png")
=======

>>>>>>> 6513c5efaf825d95b0dbf7206d8611868b7cc452
    print("FINAL EVALUATION USING THE TEST SET")
    test_results = evaluate(model, test_dataset)
    for metric_name, value in test_results.items():
        print(f"{metric_name} | {value:.4f}")

    save_model(model, "sickle_cell_edge_model.pt")
    print("The edge model has been saved as sickle_cell_edge_model.pt ")
