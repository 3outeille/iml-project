import cv2
from src.utils import *

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

SHAPE_DIMENSION = 24
SHAPE_KEYPOINTS = 200
COLOR_DIMENSION = 24
RANDOM_SEED = 42
N_NEIGHBORS = 1

def extract_color_feature_histogram(extractor, imgs):
    masks = [create_mask(img) for img in imgs]
    label_maps = [extractor.predict(img[mask]) for (img, mask) in zip(imgs, masks)]
    histograms = np.array([np.bincount(lm, minlength=len(extractor.cluster_centers_)) / len(lm) for lm in label_maps], dtype=np.float64)
    return histograms

def extract_shape_feature_histogram(extractor, imgs):
    histograms = []
    orb = cv2.ORB_create()

    for img in imgs:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keypoints = orb.detect(gray_img, None)
        keypoints, descriptors = orb.compute(gray_img, keypoints)
        bin_assignment = extractor.predict(descriptors)
        img_feats = np.zeros(SHAPE_DIMENSION)
        for id_assign in bin_assignment:
            img_feats[id_assign] += 1
        histograms.append(img_feats)

    vec = np.array(histograms)
    return vec / SHAPE_KEYPOINTS

def feature_extractor(img_train):

    def color_extractor():
        masks = [create_mask(img) for img in img_train]
        super_sample = create_super_sample(img_train, masks)
        kmean_color = KMeans(n_clusters=COLOR_DIMENSION, random_state=RANDOM_SEED)
        kmean_color.fit(super_sample)
        return kmean_color

    def shape_extractor():
        orb = cv2.ORB_create()

        list_descriptors = []

        for img in img_train:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            keypoints = orb.detect(gray_img, None)
            keypoints, descriptors = orb.compute(gray_img, keypoints)
            indices = np.random.randint(descriptors.shape[0], size=SHAPE_KEYPOINTS)
            desc_samples = descriptors[indices]
            list_descriptors.append(desc_samples)

        super_sample = np.vstack(list_descriptors)
        kmean_shape = KMeans(n_clusters=SHAPE_DIMENSION, random_state=RANDOM_SEED)
        kmean_shape.fit(super_sample)
        return kmean_shape

    kmean_color = color_extractor()
    kmean_shape = shape_extractor()

    return lambda img: extract_color_feature_histogram(kmean_color, img), lambda img: extract_shape_feature_histogram(kmean_shape, img)

def feature_extractor_fusion(color_feature_extractor, shape_feature_extractor):
    return lambda img: np.hstack((color_feature_extractor(img), shape_feature_extractor(img)))

def train(classifer, feature_histogram, label_train):
    # TODO: Cross validation ?
    classifer.fit(feature_histogram, label_train)

def evaluate(classifier, feature_histogram, label):
    prediction = classifier.predict(feature_histogram)
    accuracy = np.sum(np.array(prediction) == np.array(label)) / len(prediction)
    return prediction, accuracy

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    dataset, labels = load_dataset("./dataset/train")
    # TODO: Data augmentation ?
    img_train, img_test, label_train, label_test = train_test_split(dataset, labels, test_size=0.2, random_state=42, stratify=labels)

    print("Create feature extractor ...\n")
    color_feature_extractor, shape_feature_extractor = feature_extractor(img_train)
    
    # FIXME: Vstack seems to work as expected but investigate why it breaks on train().
    fused_feature_extractor = feature_extractor_fusion(color_feature_extractor, shape_feature_extractor)

    print("Train classifier on features ...\n")
    # classifiers = {"dummy_classifier" : DummyClassifier(), "knn_classifier": KNeighborsClassifier(n_neighbors=1)}
    # classifiers = {"dummy_classifier" : DummyClassifier()}
    classifiers = {"knn_classifier": KNeighborsClassifier(n_neighbors=N_NEIGHBORS), "dummy_classifier": DummyClassifier()}

    for name, clf in classifiers.items():
        train(clf, fused_feature_extractor(img_train), label_train)
        prediction, accuracy = evaluate(clf, fused_feature_extractor(img_test), label_test)
        print(f"accuracy = {accuracy}\n")
        dump_results(prediction, label_test, f"{name}-results")
