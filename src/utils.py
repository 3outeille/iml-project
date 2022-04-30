import cv2
import os
from matplotlib.style import available
from sklearn.utils import shuffle
import numpy as np

# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

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

def dataset_augmentation(dataset):
    def random_rotation(img):
        # pick a random degree of rotation between 180% on the left and 180% on the right
        random_degree = np.random.uniform(-180, 180)
        
        height, width = img.shape[:2]
        image_center = (width / 2, height / 2)

        rotation_mat = cv2.getRotationMatrix2D(image_center, random_degree, 1)

        radians = np.math.radians(random_degree)
        sin = np.math.sin(radians)
        cos = np.math.cos(radians)
        bound_w = int((height * abs(sin)) + (width * abs(cos)))
        bound_h = int((height * abs(cos)) + (width * abs(sin)))

        rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
        rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

        rotated_img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), borderMode = cv2.BORDER_CONSTANT, borderValue=[255, 255,255])

        (h,w,c) = rotated_img.shape
        cropped_rotated_img = []
        for i in range(h):
            if (rotated_img[i, :w, :] != 255).any():
                cropped_rotated_img.append(rotated_img[i, :w, :])
        return np.array(cropped_rotated_img)

    def random_noise(img):
        # add random noise to the image
        return np.array(sk.util.random_noise(img))

    # dictionary of the transformations we defined earlier
    available_transformations = [random_rotation, random_noise]

    dataset_aug = []
    for img in dataset:
        for transformation in available_transformations:
            dataset_aug.append(transformation(img))

    return dataset_aug

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