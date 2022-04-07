import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
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

def create_super_sample(train, masks):
    samples = []

    for (i, img) in enumerate(train):
        img_masked = img[~masks[i]]
        indices = np.random.choice(img_masked.shape[0], 50, replace=False)
        sample = img_masked[indices]

        samples.append(sample)

    super_sample = np.vstack(samples)

    return super_sample

def create_kmeans(super_sample):
    kmeans = KMeans(n_clusters=24, random_state=42)
    kmeans.fit(super_sample)
    return kmeans

if __name__ == "__main__":
    np.random.seed(42)

    train, labels = load_dataset("../dataset/train")

    masks = [create_mask(img) for img in train]
    
    super_sample = create_super_sample(train, masks)

    kmeans = create_kmeans(super_sample)

    print(kmeans.cluster_centers_)
