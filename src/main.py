from src.utils import *

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC as SVM
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from joblib import dump, load

SHAPE_DIMENSION = 48
SHAPE_KEYPOINTS = 96
COLOR_DIMENSION = 24
PONDERATION = 0
RANDOM_SEED = 42
N_NEIGHBORS = 3
N_DEPTH = 8
MODE = 'orb'
SAVE_SESSION = False

def extract_color_feature_histogram(extractor, imgs):
    masks = [create_mask(img) for img in imgs]
    label_maps = [extractor.predict(img[mask]) for (img, mask) in zip(imgs, masks)]
    histograms = np.array([np.bincount(lm, minlength=len(extractor.cluster_centers_)) / len(lm) for lm in label_maps], dtype=np.float64)
    return histograms

def extract_shape_feature_moments(imgs):
    res = []

    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.Canny(img, 100, 200)

        moments = cv2.moments(img)

        hu_moments = cv2.HuMoments(moments).flatten()

        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

        res.append(hu_moments)

    res = np.array(res)

    return res

def extract_shape_feature_histogram(extractor, imgs):
    histograms = []
    orb = cv2.ORB_create()

    for img in imgs:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keypoints = orb.detect(gray_img, None)
        if keypoints == ():
            histograms.append(np.zeros(SHAPE_DIMENSION))
            continue
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
            if keypoints == ():
                continue
            keypoints, descriptors = orb.compute(gray_img, keypoints)
            indices = np.random.randint(descriptors.shape[0], size=SHAPE_KEYPOINTS)
            desc_samples = descriptors[indices]
            list_descriptors.append(desc_samples)

        super_sample = np.vstack(list_descriptors)
        kmean_shape = KMeans(n_clusters=SHAPE_DIMENSION, random_state=RANDOM_SEED)
        kmean_shape.fit(super_sample)
        return kmean_shape

    kmean_color = color_extractor()
    if (MODE == 'orb'):
        kmean_shape = shape_extractor()
        return lambda img: extract_color_feature_histogram(kmean_color, img), lambda img: extract_shape_feature_histogram(kmean_shape, img)
    else:
        return lambda img: extract_color_feature_histogram(kmean_color, img), lambda img: extract_shape_feature_moments(img)

def feature_extractor_fusion(color_feature_extractor, shape_feature_extractor):
    return lambda img: np.hstack((color_feature_extractor(img) * PONDERATION, shape_feature_extractor(img) * (1 - PONDERATION)))

def train(classifer, feature_histogram, labels_train_aug):
    # TODO: Standard scaler?
    classifer.fit(feature_histogram, labels_train_aug)

def evaluate(classifier, feature_histogram, label):
    prediction = classifier.predict(feature_histogram)
    accuracy = np.sum(np.array(prediction) == np.array(label)) / len(prediction)
    return prediction, accuracy

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    dataset, labels = load_dataset("./dataset/train")
    img_train, img_test, label_train, label_test = train_test_split(dataset, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels)

    img_train_aug, labels_train_aug = dataset_augmentation(img_train, label_train)
    
    print("Create feature extractor ...\n")
    color_feature_extractor, shape_feature_extractor = feature_extractor(img_train_aug)

    fused_feature_extractor = feature_extractor_fusion(color_feature_extractor, shape_feature_extractor)

    print("Train classifier on features ...\n")

    classifiers = {
        "dummy_classifier": DummyClassifier(), #TODO: REMOVE ME
        "knn_classifier": KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric='cosine'),
        "svm_classifier_linear": SVM(kernel='linear'),
        "svm_classifier_polynomial": SVM(kernel='poly'),
        "svm_classifier_rbf": SVM(kernel='rbf'),
        "random_forest_classifier": RandomForestClassifier(max_depth=N_DEPTH)
    }

    for name, clf in classifiers.items():
        train(clf, fused_feature_extractor(img_train_aug), labels_train_aug)
        prediction, accuracy = evaluate(clf, fused_feature_extractor(img_test), label_test)
        print(f"{name} accuracy = {accuracy}\n")
        # dump_results(prediction, label_test, f"{name}-results")

    #TODO: save weights of classifier & kmeans extractor
    #TODO: integrate early/late fusion + stacking