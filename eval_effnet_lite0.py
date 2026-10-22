import tensorflow as tf
import keras_hub #for keras to recognize the EfficientNet layers
from step2_data_pipeline import get_datasets, efficientnet_preprocess #from data pipleine

#file variable to contain the fine tuned model of EfficientNet lite0
MODEL_FILE = "sickle_cell_effnet_lite0_ft30.keras"

#load only test images, using EfficientNet preprocessing
_, _, test_dataset, _ = get_datasets(efficientnet_preprocess)

#load saved model
model = tf.keras.models.load_model(MODEL_FILE)
#gGet model's confidence (0 to 1) for every test image
probs = model.predict(test_dataset).ravel()

#Collect the true labels in the same order since they aren't shuffled
labels = tf.concat([y for _, y in test_dataset], axis=0).numpy().ravel().astype(int)
#turn the confidence into a decision: 0.5 or above to mean sickle
preds = (probs >= 0.5).astype(int)

#print the confusion matrix
print(MODEL_FILE)
print(tf.math.confusion_matrix(labels, preds).numpy())