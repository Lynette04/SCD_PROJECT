import tensorflow as tf                                                    # loads the models
import keras_hub                                                           # needed so Keras can load the EfficientNet models
from step2_data_pipeline import get_datasets, efficientnet_preprocess      # your data pipeline

# Each entry: a name for the printout, the saved model file, and the preprocessing that model was trained with
# (None means the default MobileNetV2 scaling)
MODELS = [
    ("MobileNetV2 frozen",        "sickle_cell_model.keras",             None),
    ("MobileNetV2 fine-tuned",    "Sickle_cell_fine_tuned-mode.keras",   None),
    ("MobileNetV2 edge",          "sickle_cell_edge_model.keras",        None),
    ("EfficientNet-Lite0 frozen", "sickle_cell_effnet_lite0.keras",      efficientnet_preprocess),
    ("EfficientNet-Lite0 ft30",   "sickle_cell_effnet_lite0_ft30.keras", efficientnet_preprocess),
]

for name, file, prep in MODELS:                       # handle one model at a time
    if prep is None:
        _, _, test_dataset, _ = get_datasets()        # default MobileNetV2 preprocessing
    else:
        _, _, test_dataset, _ = get_datasets(prep)    # EfficientNet preprocessing

    model = tf.keras.models.load_model(file)          # load the saved model

    probs = model.predict(test_dataset, verbose=0).ravel()   # the model's confidence (0 to 1) for each test image
    labels = tf.concat([y for _, y in test_dataset], axis=0).numpy().ravel().astype(int)  # true labels, same order
    preds = (probs >= 0.5).astype(int)                # 0.5 or above means sickle

    cm = tf.math.confusion_matrix(labels, preds).numpy()     # rows = true class, columns = predicted class
    tn, fp, fn, tp = cm.ravel()                       # unpack the four counts

    print("\n" + name)
    print(cm)
    print(f"accuracy {(tp + tn) / cm.sum():.3f} | precision {tp / (tp + fp):.3f} | recall {tp / (tp + fn):.3f}")