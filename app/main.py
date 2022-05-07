from utils import *
from fusion import *

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pickle

SHAPE_DIMENSION = 48
SHAPE_KEYPOINTS = 96
COLOR_DIMENSION = 24
PONDERATION = 0.4
RANDOM_SEED = 40
N_NEIGHBORS = 3
N_DEPTH = 8
TRANSFORMATION_PER_IMG = 0
N_ESTIMATORS = 10
CROSS_VALIDATION = 4
MODE = 'moments'
LOAD_SESSION = False


def extract_color_feature_histogram(extractor, imgs):
    masks = [create_mask(img) for img in imgs]
    label_maps = [extractor.predict(img[mask])
                  for (img, mask) in zip(imgs, masks)]
    histograms = np.array([np.bincount(lm, minlength=len(
        extractor.cluster_centers_)) / len(lm) for lm in label_maps], dtype=np.float64)
    return histograms


def extract_shape_feature_moments(imgs):
    res = []

    for i, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        height, width = img.shape
        if height > width:
            img = cv2.copyMakeBorder(
                img, 0, 0, (height - width) // 2, (height - width) // 2, cv2.BORDER_CONSTANT, value=255)
        elif width > height:
            img = cv2.copyMakeBorder(
                img, (width - height) // 2, (width - height) // 2, 0, 0,  cv2.BORDER_CONSTANT, value=255)
        
        img = cv2.resize(img, (100, 100))

        img = cv2.copyMakeBorder(
            img, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)

        img = cv2.GaussianBlur(img, (5, 5), 0)
        _, img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.Canny(img, 100, 200)

        # plt.imsave(f'test/{i}.png', img)

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


def feature_extractor(img_train, load_session=False, mode="moments"):

    cwd = os.getcwd()
    # During developpement
    if cwd != "/app":
        cwd += "/app"

    def color_extractor():
        if load_session == False:
            masks = [create_mask(img) for img in img_train]
            super_sample = create_super_sample(img_train, masks)
            kmean_color = KMeans(n_clusters=COLOR_DIMENSION,
                                 random_state=RANDOM_SEED)
            kmean_color.fit(super_sample)
            pickle.dump(kmean_color, open(f"{cwd}/kmean_color.pkl", "wb"))
        else:
            kmean_color = pickle.load(open(f"{cwd}/kmean_color.pkl", "rb"))

        return kmean_color

    def shape_extractor():

        if load_session == False:
            orb = cv2.ORB_create()

            list_descriptors = []

            for img in img_train:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                keypoints = orb.detect(gray_img, None)
                if keypoints == ():
                    continue
                keypoints, descriptors = orb.compute(gray_img, keypoints)
                indices = np.random.randint(
                    descriptors.shape[0], size=SHAPE_KEYPOINTS)
                desc_samples = descriptors[indices]
                list_descriptors.append(desc_samples)

            super_sample = np.vstack(list_descriptors)
            kmean_shape = KMeans(n_clusters=SHAPE_DIMENSION,
                                 random_state=RANDOM_SEED)
            kmean_shape.fit(super_sample)
            pickle.dump(kmean_shape, open(f"{cwd}/kmean_shape.pkl", "wb"))
        else:
            kmean_shape = pickle.load(open(f"{cwd}/kmean_shape.pkl", "rb"))

        return kmean_shape

    kmean_color = color_extractor()

    if (mode == 'orb'):
        print("Mode ORB ...")
        kmean_shape = shape_extractor()
        return lambda img: extract_color_feature_histogram(kmean_color, img), lambda img: extract_shape_feature_histogram(kmean_shape, img)
    else:
        print("Mode Hu moments ...")
        return lambda img: extract_color_feature_histogram(kmean_color, img), lambda img: extract_shape_feature_moments(img)


import time
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    print('Loading dataset...')
    dataset, labels = load_dataset("./dataset/train")
    print('Dataset augmentation...')
    dataset_aug, labels_aug = dataset_augmentation(
        dataset, labels, TRANSFORMATION_PER_IMG)

    img_train, img_test, label_train, label_test = train_test_split(
        dataset_aug, labels_aug, test_size=0.2, random_state=RANDOM_SEED, stratify=labels_aug)

    print("Create feature extractor ...\n")
    color_feature_extractor, shape_feature_extractor = feature_extractor(
        img_train, load_session=LOAD_SESSION, mode=MODE)

    early_fusion(img_train, label_train, img_test, label_test,
                 color_feature_extractor, shape_feature_extractor, n_neighbors=N_NEIGHBORS, n_depth=N_DEPTH, ponderation=PONDERATION)

    late_fusion(img_train, label_train, img_test, label_test,
                color_feature_extractor, shape_feature_extractor, n_neighbors=N_NEIGHBORS, n_depth=N_DEPTH)

    stacking_early_fusion(img_train, label_train, img_test, label_test, color_feature_extractor,
                             shape_feature_extractor, random_seed=RANDOM_SEED, n_estimators=N_ESTIMATORS, cv=CROSS_VALIDATION, ponderation=PONDERATION)

    stacking_late_fusion(img_train, label_train, img_test, label_test, color_feature_extractor,
                             shape_feature_extractor, random_seed=RANDOM_SEED, n_estimators=N_ESTIMATORS, cv=CROSS_VALIDATION, ponderation=PONDERATION)
