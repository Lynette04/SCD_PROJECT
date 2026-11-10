import numpy as np
import torch

from pytorch_training import load_model, predict                         # loads the models and gets their predictions
from step2_data_pipeline import get_datasets, efficientnet_preprocess      # your data pipeline

# Each entry: a name for the printout, the saved model file, and the preprocessing that model was trained with
# (None means the default MobileNetV2 scaling)
MODELS = [
    ("MobileNetV2 frozen",        "sickle_cell_model.pt",             None),
    ("MobileNetV2 fine-tuned",    "Sickle_cell_fine_tuned-mode.pt",   None),
    ("MobileNetV2 edge",          "sickle_cell_edge_model.pt",        None),
    ("EfficientNet-Lite0 frozen", "sickle_cell_effnet_lite0.pt",      efficientnet_preprocess),
    ("EfficientNet-Lite0 ft30",   "sickle_cell_effnet_lite0_ft30.pt", efficientnet_preprocess),
]

for name, file, prep in MODELS:                       # handle one model at a time
    if prep is None:
        _, _, test_dataset, _ = get_datasets()        # default MobileNetV2 preprocessing
    else:
        _, _, test_dataset, _ = get_datasets(prep)    # EfficientNet preprocessing

    model = load_model(file)                          # load the saved model

    probs = predict(model, test_dataset)              # the model's confidence (0 to 1) for each test image
    labels = torch.cat([y for _, y in test_dataset]).numpy().ravel().astype(int)  # true labels, same order
    preds = (probs >= 0.5).astype(int)                # 0.5 or above means sickle

    cm = np.zeros((2, 2), dtype=int)                  # rows = true class, columns = predicted class
    np.add.at(cm, (labels, preds), 1)
    tn, fp, fn, tp = cm.ravel()                       # unpack the four counts

    print("\n" + name)
    print(cm)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp +fn) if (tp+fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"accuracy {(tp + tn) / cm.sum():.3f} | precision {precision:.3f} | recall {recall:.3f} | f1{f1:.3f}")
    
