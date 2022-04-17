import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
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
    return ~selector

def create_super_sample(train, masks):
    samples = []

    for (i, img) in enumerate(train):
        img_masked = img[masks[i]]
        indices = np.random.choice(img_masked.shape[0], 50, replace=False)
        sample = img_masked[indices]

        samples.append(sample)

    super_sample = np.vstack(samples)

    return super_sample

def create_kmeans(super_sample):
    kmeans = KMeans(n_clusters=24, random_state=42)
    kmeans.fit(super_sample)
    return kmeans

# def quantize_img_color(kmeans, img, mask, color_lut):
#     label_map = kmeans.predict(img[mask])
#     recolored_img = np.ones_like(img) * 255
#     recolored_img[mask] = color_lut[label_map]
#     return recolored_img

def main(dataset_path):
    np.random.seed(42)

    train, labels = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=42, stratify=labels)

    masks_train = [create_mask(img) for img in X_train]
    masks_test = [create_mask(img) for img in X_test]

    super_sample = create_super_sample(X_train, masks_train)

    kmeans = create_kmeans(super_sample)

    # color_lut = np.uint8(kmeans.cluster_centers_)
    labels_maps_train = [kmeans.predict(img[mask]) for (img, mask) in zip(X_train, masks_train)]
    color_histograms_train = np.array([np.bincount(lm, minlength=len(kmeans.cluster_centers_)) / len(lm) for lm in labels_maps_train], dtype=np.float64)
    
    labels_maps_test = [kmeans.predict(img[mask]) for (img, mask) in zip(X_test, masks_test)]
    color_histograms_test = np.array([np.bincount(lm, minlength=len(kmeans.cluster_centers_)) / len(lm) for lm in labels_maps_test], dtype=np.float64)
    
    dist_mat = cdist(color_histograms_test, color_histograms_train, metric='cosine')
    


if __name__ == "__main__":
    main("../dataset/train")
    