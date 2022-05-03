from src.utils import *

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC as SVM
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process.kernels import RBF
from joblib import dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB


SHAPE_DIMENSION = 48
SHAPE_KEYPOINTS = 96
COLOR_DIMENSION = 24
PONDERATION = 0
RANDOM_SEED = 42
N_NEIGHBORS = 3
N_DEPTH = 8
MODE = 'moment'
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

def get_accuracy(prediction, label):
    accuracy = np.sum(np.array(prediction) == np.array(label)) / len(prediction)
    return accuracy

def evaluate(classifier, feature_histogram, label):
    prediction = classifier.predict(feature_histogram)
    accuracy = get_accuracy(prediction, label)
    return prediction, accuracy

# Early fusion
def early_fusion(img_train_aug, labels_train_aug, img_test, label_test, color_feature_extractor, shape_feature_extractor):
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

# Late fusion
def get_max_proba_indices(color_predict_proba, shape_predict_proba):
    classes_pred_proba = 0.6 * color_predict_proba + 0.4 * shape_predict_proba
    return np.argmax(classes_pred_proba, axis=1)

def soft_voting(color_predict_proba, shape_predict_proba, color_labels, shape_labels):
    max_proba_indces = get_max_proba_indices(color_predict_proba, shape_predict_proba)
    
    # TODO stop cheating
    return max_proba_indces + 1

def late_fusion(img_train, label_train, img_test, label_test, color_feature_extractor, shape_feature_extractor):
    classifiers = {"knn_clf-knn_clf": [KNeighborsClassifier(n_neighbors=N_NEIGHBORS), KNeighborsClassifier(n_neighbors=N_NEIGHBORS)],
                   "dummy_clf-dummy_clf": [DummyClassifier(), DummyClassifier()]}

    for name, clf in classifiers.items():
        print(f"Train {name} classifier on color features ...\n")
        train(clf[0], color_feature_extractor(img_train), label_train)
        print(f"Train {name} classifier on shape features ...\n")
        train(clf[1], shape_feature_extractor(img_train), label_train)

        color_labels = clf[0].predict(color_feature_extractor(img_test))
        shape_labels = clf[1].predict(shape_feature_extractor(img_test))
        prediction = soft_voting(clf[0].predict_proba(color_feature_extractor(img_test)), clf[1].predict_proba(shape_feature_extractor(img_test)), color_labels, shape_labels)

        accuracy = get_accuracy(prediction, label_test)
        print(f"accuracy = {accuracy}\n")
        dump_results(prediction, label_test, f"{name}-results")

# Fusion with stacking classifier
def fusion_with_stacking_clf(img_train, label_train, img_test, label_test, color_feature_extractor, shape_feature_extractor):
    fused_feature_extractor = feature_extractor_fusion(color_feature_extractor, shape_feature_extractor)

    def base_model():
        level0 = list()
        level0.append(('svm', SVM()))
        level0.append(('cart', DecisionTreeClassifier()))
        level0.append(('rf', RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)))
        level0.append(('svr', LinearSVC(random_state=RANDOM_SEED)))

        model = StackingClassifier(estimators=level0, final_estimator=LogisticRegression(), cv=40)
        return model

    def model_with_pipeline():
        level0 = list()
        # level0.append(('svm', SVM()))
        level0.append(('cart', DecisionTreeClassifier()))
        # level0.append(('rf', RandomForestClassifier(n_estimators=10, random_state=42)))
        level0.append(('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=RANDOM_SEED, max_iter=10000))))
        model = StackingClassifier(estimators=level0, final_estimator=LogisticRegression(), cv=40)
        return model
    
    stacking_clf = base_model()
    train(stacking_clf, fused_feature_extractor(img_train), label_train)
    prediction, accuracy = evaluate(stacking_clf, fused_feature_extractor(img_test), label_test)
    print(f"stacking clf accuracy = {accuracy}\n")
    # dump_results(prediction, label_test, f"results")
    
    stacking_clf_with_pipeline = model_with_pipeline()
    train(stacking_clf_with_pipeline, fused_feature_extractor(img_train), label_train)
    prediction, accuracy = evaluate(stacking_clf_with_pipeline, fused_feature_extractor(img_test), label_test)
    print(f"stacking clf with pipeline accuracy = {accuracy}\n")
    # dump_results(prediction, label_test, f"results")

    # Fusion with voting classifier
def fusion_with_voting_clf(img_train, label_train, img_test, label_test, color_feature_extractor, shape_feature_extractor):
    fused_feature_extractor = feature_extractor_fusion(color_feature_extractor, shape_feature_extractor)

    def make_model(voting_type):
        estimators = list()
        estimators.append(('cart', DecisionTreeClassifier()))
        estimators.append(('rf', RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)))
        estimators.append(('gnb', GaussianNB()))
        if voting_type == 'hard':
            estimators.append(('lr', LogisticRegression(multi_class='multinomial', random_state=1)))
            estimators.append(('svm', SVM()))
        model = VotingClassifier(estimators=estimators, voting=voting_type)
        return model

    def model_with_pipeline(voting_type):
        estimators = list()
        # estimators.append(('cart', DecisionTreeClassifier()))
        estimators.append(('rf', RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)))
        if voting_type == 'hard':
            estimators.append(('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42))))
        else:
            estimators.append(('srf', make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=10, random_state=42))))
        model = VotingClassifier(estimators=estimators, voting=voting_type)
        return model
    
    voting_type = 'soft'
    voting_clf = make_model(voting_type)
    # voting_clf = model_with_pipeline(voting_type)
    train(voting_clf, fused_feature_extractor(img_train), label_train)
    prediction, accuracy = evaluate(voting_clf, fused_feature_extractor(img_test), label_test)
    print(f"{voting_type} voting clf accuracy = {accuracy}\n")
    # dump_results(prediction, label_test, f"results")

# def scale_data(img_train, img_test):
#     scaler = StandardScaler()
#     scaled_train = scaler.fit_transform(img_train)
#     scaled_test = scaler.transform(img_test)
#     return scaled_train, scaled_test

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    dataset, labels = load_dataset("./dataset/train")
    img_train, img_test, label_train, label_test = train_test_split(dataset, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels)

    img_train_aug, labels_train_aug = dataset_augmentation(img_train, label_train)

    # TODO implement data scaling, solve 'ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2508,) + inhomogeneous part.'
    # img_train_aug, img_test = scale_data(img_train_aug, img_test)
    
    print("Create feature extractor ...\n")
    color_feature_extractor, shape_feature_extractor = feature_extractor(img_train_aug)

    # Early fusion
    # early_fusion(img_train_aug, labels_train_aug, img_test, label_test, color_feature_extractor, shape_feature_extractor)
    
    # Late fusion
    # late_fusion(img_train_aug, labels_train_aug, img_test, label_test, color_feature_extractor, shape_feature_extractor)
    
    # Fusion with stacking_clf
    # fusion_with_stacking_clf(img_train_aug, labels_train_aug, img_test, label_test, color_feature_extractor, shape_feature_extractor)

    # Fusion with voting_clf
    fusion_with_voting_clf(img_train_aug, labels_train_aug, img_test, label_test, color_feature_extractor, shape_feature_extractor)

