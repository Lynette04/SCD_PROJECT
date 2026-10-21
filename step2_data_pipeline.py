import tensorflow as tf #module that can teachg AI
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input #type: ignore
import os #to count files in folders

img_size = (224,224) #224 pixels tall and 224 pixels wide
batch_size = 16 #16 images in a batch

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406]) * 255.0
IMAGENET_STD = tf.constant([0.229,0.224,0.225]) * 255.0

def efficientnet_preprocess(images):
    return (images - IMAGENET_MEAN) / IMAGENET_STD

#create a sort of pipeline with ordered steps(sequential) to perform on the training set images
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"), #turn the images in opposite directions like right/left or up/down
    tf.keras.layers.RandomRotation(0.1), #rotate the image by 10% of a full turn
    tf.keras.layers.RandomBrightness(0.1), #randomize the brightness by 0.1
    tf.keras.layers.RandomContrast(0.1) #randomize the contrast by 0.1
])


def get_datasets(preprocess_fn=preprocess_input):
    #check in the train folder inside data and resize each image to 224 by 224 pixels  
    #and group the images into groups of 16 called batches. 
    #The labels should be in binary 0 for Normal and 1 for sickle
    training_dataset = tf.keras.utils.image_dataset_from_directory(
        "data/train", 
        image_size= img_size,
        batch_size = batch_size,
        label_mode= "binary",   
    )
    #the same is done for the val and test folders but the files shouldnt be shuffled since the order doesnt matter for evaluation.
    validation_dataset= tf.keras.utils.image_dataset_from_directory(
        "data/val",
        image_size= img_size,
        batch_size = batch_size,
        label_mode= "binary",
        shuffle=False
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        "data/test",
        image_size= img_size,
        batch_size = batch_size,
        label_mode= "binary",
        shuffle=False
    )
    
    #Store the class names of the training dataset in its own variable claas_names
    class_names= training_dataset.class_names
    
    #Function that takes a batch of images with their labels, augments then normalizes them and finally returns the images with their labels.
    def prepare_train_batch(images,labels):
        images = data_augmentation(images)
        images = preprocess_fn(images)
        return images, labels
    #Similar function to above but has no augmentation since the other images are not for training the model
    def prepare_evaluation_batch(images, labels):
        images= preprocess_fn(images)
        return images, labels
    
    #Run the datasets through their required functions
    training_dataset = training_dataset.map(prepare_train_batch)
    validation_dataset = validation_dataset.map(prepare_evaluation_batch)
    test_dataset = test_dataset.map(prepare_evaluation_batch)
    
    
    
    return training_dataset, validation_dataset, test_dataset, class_names

def get_class_weights(): #function to calculate class weights
    train_path = "data/train" #path to count files
    class_counts = {} #dictionary to store the number of images in each class of normal and sickle.

    #for loop to count the number of files in each class and add then add the number to the dictionary of class counts.
    #the loop has to get the class names in alphabetical order(normal then sickle) to match the keras order of 0-normal and 1-sickle
    for class_name in sorted(os.listdir(train_path)):
        class_folder = os.path.join(train_path, class_name)
        class_counts[class_name] = len(os.listdir(class_folder))


    
    #to solve the imbalance of normal images being fewer than the sickle images
    #let's introduce class weights so that each class is multiplied by its weight.
    #We don't want the model to become lazy and wrongly classify normal images as sickle images. It has to pay more attention to the normal since they are the minority in number.

    total_number_of_images = sum(class_counts.values()) #sum of images in normal and sickle folders
    number_of_classes = len(class_counts) # 2 in number: normal and sickle

    class_weights= {} #dictionary to hold the weights for each class
    sorted_class_names = sorted(class_counts.keys()) #alphabetical order for the classes

    for index, class_name in enumerate(sorted_class_names): #enumerate gives both the index and the item in that position.
        #class with fewer images gets a bigger weight so mistakes are amplified and model has to work on reducing mistakes
        class_weights[index] = total_number_of_images / (number_of_classes * class_counts[class_name])  
    
    return class_weights, class_counts

#Testing block of code to see that the functions work.
if __name__ == "__main__":
    training_dataset, validation_dataset, test_dataset, class_names = get_datasets()
    print("Class names:", class_names)
    
    class_weights, class_counts = get_class_weights()
    print("Training image counts:", class_counts)
    print("Class weights:", class_weights)

    #get one batch and split in into images and the matching labels
    for images, labels in training_dataset.take(1): 
        print("Shape of one batch of images:", images.shape) #Expected output is (16,224,224,3) 3: each pixel has 3 color values RGB
        print("Pixel value raange after preprocessing:", float(tf.reduce_min(images)), "to", float(tf.reduce_max(images)))
        
    