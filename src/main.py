import cv2
import numpy as np
import os

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

    return train, labels

if __name__ == "__main__":
    train, labels = load_dataset("../dataset/train")
