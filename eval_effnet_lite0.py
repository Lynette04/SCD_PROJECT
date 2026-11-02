import numpy as np
import torch

from pytorch_training import load_model, predict
from step2_data_pipeline import get_datasets, efficientnet_preprocess #from data pipleine

#file variable to contain the fine tuned model of EfficientNet lite0
MODEL_FILE = "sickle_cell_effnet_lite0_ft30.pt"

#load only test images, using EfficientNet preprocessing
_, _, test_dataset, _ = get_datasets(efficientnet_preprocess)

#load saved model
model = load_model(MODEL_FILE)
#gGet model's confidence (0 to 1) for every test image
probs = predict(model, test_dataset)

#Collect the true labels in the same order since they aren't shuffled
labels = torch.cat([y for _, y in test_dataset]).numpy().ravel().astype(int)
#turn the confidence into a decision: 0.5 or above to mean sickle
preds = (probs >= 0.5).astype(int)

#print the confusion matrix, each row is the true class and each column is the predicted class
confusion_matrix = np.zeros((2, 2), dtype=int)
np.add.at(confusion_matrix, (labels, preds), 1)
print(MODEL_FILE)
print(confusion_matrix)
