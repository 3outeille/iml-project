import cv2
import numpy as np
import os
from sklearn.utils import shuffle

def load_dataset(path):
    train, labels = [], []
    
    for directory in os.listdir(path):
        if directory == '.DS_Store':
            continue
        for filename in os.listdir(os.path.join(path, directory)):
            label = int(directory)

            complete_path = os.path.join(path, directory, filename)

            raw = cv2.imread(complete_path)
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

            train.append(img)
            labels.append(label)

    return shuffle(train, labels)

def create_mask(img):
    selector = (img[...,0] == 255) & (img[...,1] == 255) & (img[...,2] == 255)
    return selector

if __name__ == "__main__":
    train, labels = load_dataset("../dataset/train")
    

    masks = [create_mask(img) for img in train]
    

