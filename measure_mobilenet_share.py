import tensorflow as tf 
from tensorflow.keras import layers #type: ignore layers to help us check the layers of BatchNormalisation.  

#loading the frozen MobileNetV2 model, no training will happen here
model = tf.keras.models.load_model("sickle_cell_model.keras")

#Position 1 is the MobileNetV2 backbone
base_model = model.layers[1]

#Since the backbone is saved in a file, it needs to be switched on first otherwise keras reports zero trainable weights.
base_model.trainable = True

#Using only last 30 layers like in the fine tuning script for the second model variant of mobilenetv2
start = len(base_model.layers) - 30
for i, lr in enumerate(base_model.layers):
    if i < start or isinstance(lr, layers.BatchNormalization):
        lr.trainable = False
    else:
        lr.trainable = True
        
#Sum up the weights that would train inside the backbone, and compare with the backbone's total

trainable = sum(w.numpy().size for w in base_model.trainable_weights)
total = base_model.count_params()
print(f"Backbone trainable weights:{trainable} of {total} = {round(100 * trainable / total, 1)} percent")


