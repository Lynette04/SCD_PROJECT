import os #module to check folder contents,create folders, chek if files exist
import random #module to randomize things
import shutil #module for file operations like copying, moving and deleting.

random.seed(42) #to get the same shuffle every time the code is run.

#Create a dictionary to store the paths for the negatives and positives
class_folders={
    "normal": r"C:\Users\kaboh\OneDrive\Desktop\Project prototype\Sickle Cell disease dataset\Negative\Clear",
    "sickle": r"C:\Users\kaboh\OneDrive\Desktop\Project prototype\Sickle Cell disease dataset\Positive\Unlabelled"
    }
output_root = "data" #variable for folder where the organised copies of images will go

#Create a for loop which is able to: -calculate the number of files in each class
#-split the files into training, validating and testing files
#- organise the files into their different destination folders from their original folders
for class_name, folder_path in class_folders.items():
    
    files = os.listdir(folder_path) #list of every file in the folder
    random.shuffle(files) #shuffle the files based on the random seed
    
    total=len(files)
    n_train = int(total * 0.70) #70% of images are used for training
    n_val = int(total * 0.15) #15% of images are used for validation and remainder is for testing.
    
    #section the shuffled lists into 3 different groups
    train_files = files[:n_train]  # e.g [1: 100]
    val_files = files[n_train:n_train + n_val] #e.g[100: 200]
    test_files =files[n_train + n_val:] #e.g[200:last]
    
    #dictionary to hold the 3 groups together
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    for split_name, file_list in splits.items():
        #create folder path for groups
        destination_folder = os.path.join(output_root, split_name, class_name)   # e.g data/train/normal
        os.makedirs(destination_folder, exist_ok=True) #create the folder and dont bring error if folder already exists
        
        for filename in file_list:
            #current location of a file
            source_path = os.path.join(folder_path,filename) # e.g C:\Users\kaboh\OneDrive\Desktop\Project prototype\Sickle Cell disease dataset\Negative\Clear\1.jpg
            #destination location of a file
            destination_path = os.path.join(destination_folder,filename)  #e.g "C:\Users\kaboh\OneDrive\Desktop\Project prototype\scd_trial\data\train\normal\4.jpg"
            shutil.copy2(source_path,destination_path) #copy file to its destination
    
    print("The ",class_name, "group is done copying training: ",len(train_files),"files, validating files: ",len(val_files), "files and testing files: ",len(test_files)," files.")
    




