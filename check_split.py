import os #module for working with folders and file names

files = {} #an empty dictionary that will hold the file names of the six folders
for s in ["train", "val", "test"]: #go through the three splits one at a time
    for c in ["normal", "sickle"]: #and, inside each split, the two classes
        # os.path.join builds a path such as data/train/normal
        # os.listdir lists every file name inside that folder
        # set() turns the list into a set, which is needed for the overlap check below
        names = set(os.listdir(os.path.join("data", s, c)))
        files[(s,c)] = names # store the names under a key such as ("train", "normal")
        print(s,c,len(names)) # print the split, the class and how many files it holds
        
for c in ["normal", "sickle"]:  # check each class separately
    for a,b in [("train","val"), ("train","test"),("val","test")]:
        # & keeps only the file names that appear in both sets
        # len() counts them, so 0 means no image is shared between the two splits
        print("overlap", c, a, b, len(files[(a,c)] & files[(b,c)]))
        
        