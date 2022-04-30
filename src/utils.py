import cv2
import os
from sklearn.utils import shuffle
import numpy as np

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

    return np.vstack(samples)