import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

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

def extract_histograms_color(kmeans, imgs):
    masks = [create_mask(img) for img in imgs]

    label_maps = [kmeans.predict(img[mask]) for (img, mask) in zip(imgs, masks)]
    histograms = np.array([np.bincount(lm, minlength=len(kmeans.cluster_centers_)) / len(lm) for lm in label_maps], dtype=np.float64)

    return histograms
    
def create_model(img_train, label_train):

    def color_extractor():
        masks = [create_mask(img) for img in img_train]
        super_sample = create_super_sample(img_train, masks)

        print('Training Kmeans')
        kmeans = create_kmeans(super_sample)
        return kmeans

    def shape_extractor():
        orb = cv2.ORB_create()

        list_descriptors = []

        for img in img_train:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            keypoints = orb.detect(gray_img, None)
            keypoints, descriptors = orb.compute(gray_img, keypoints)

            indices = np.random.choice(descriptors.shape[0], 50, replace=False)
            desc_samples = descriptors[indices]
            list_descriptors.append(desc_samples)

        super_sample = np.vstack(list_descriptors)
        kmeans = create_kmeans(super_sample)
        return kmeans
        
    kmeans_shape = shape_extractor()
    kmeans_color = color_extractor()

    print('Creating Color histograms')
    color_histograms_train = extract_histograms_color(kmeans_color, img_train)

    # print('Creating Shape histograms')
    # shape_histograms_train = extract_histograms_shape(kmeans_shape, img_train)

    # concat histograms of color + shape:
    histograms_train = color_histograms_train #+ shape_histograms_train

    clf = KNeighborsClassifier(n_neighbors=1)
    # clf = DummyClassifier()
    clf.fit(histograms_train, label_train)

    def model(img_test):
        histograms_test = extract_histograms_color(kmeans_color, img_test)
        # shape_test = extract_shape(kmeans_shape, img_test)

        # concat histograms + shape
        
        # dist_mat = cdist(histograms_test, histograms_train, metric='cosine')
        # idx_of_best_matches_per_row = np.argsort(dist_mat, axis=1)

        # indexes = idx_of_best_matches_per_row[:, 0]
        # return np.array(label_train)[indexes]
        return clf.predict(histograms_test)

    return model


# def quantize_img_color(kmeans, img, mask, color_lut):
#     label_map = kmeans.predict(img[mask])
#     recolored_img = np.ones_like(img) * 255
#     recolored_img[mask] = color_lut[label_map]
#     return recolored_img

def load_test_images(path):
    img_test, filenames = [], []

    for filename in os.listdir(path):
        complete_path = os.path.join(path, filename)

        raw = cv2.imread(complete_path)
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        img_test.append(img)
        filenames.append(filename)

    return img_test, filenames

def dump_results(results, filenames_test, path):
    with open(path, 'w') as file:
        for filename, predicted_class_id in zip(filenames_test, results):
            line = f'{filename},{predicted_class_id}\n'
            file.write(line)

def live(dataset_path, test_path):
    np.random.seed(42)

    img_train, label_train = load_dataset(dataset_path)

    model = create_model(img_train, label_train)

    img_test, filenames_test = load_test_images(test_path)
    results = model(img_test)

    dump_results(results, filenames_test, "results.txt")

def accuracy_test(dataset_path):
    np.random.seed(42)

    train, labels = load_dataset(dataset_path)
    img_train, img_test, label_train, label_test = train_test_split(train, labels, test_size=0.2, random_state=42, stratify=labels)

    model = create_model(img_train, label_train)

    results = model(img_test)

    accuracy = np.sum(np.array(results) == np.array(label_test)) / len(results)

    print("accuracy:", accuracy)

def main(dataset_path, test_path):
    # live(dataset_path, test_path)
    accuracy_test(dataset_path)

if __name__ == "__main__":
    main("../dataset/train", "../tests")