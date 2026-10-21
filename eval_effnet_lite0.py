import tensorflow as tf
import keras_hub
from step2_data_pipeline import get_datasets, efficientnet_preprocess

_, _, test_dataset, _ = get_datasets(efficientnet_preprocess)
model = tf.keras.models.load_model("sickle_cell_effnet_lite0.keras")

probs = model.predict(test_dataset).ravel()
labels = tf.concat([y for _, y in test_dataset], axis=0).numpy().ravel().astype(int)
preds = (probs >= 0.5).astype(int)
print(tf.math.confusion_matrix(labels, preds).numpy())