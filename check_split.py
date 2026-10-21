import os

files = {}
for s in ["train", "val", "test"]:
    for c in ["normal", "sickle"]:
        names = set(os.listdir(os.path.join("data", s, c)))
        files[(s,c)] = names
        print(s,c,len(names))
        
for c in ["normal", "sickle"]:
    for a,b in [("train","val"), ("train","test"),("val","test")]:
        print("overlap", c, a, b, len(files[(a,c)] & files[(b,c)]))
        
        