import cv2
import os
from matplotlib.style import available
from sklearn.utils import shuffle
import numpy as np
import copy

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

def dataset_augmentation(dataset, labels, transformation_per_image=5):
    def random_rotation(img):
        # pick a random degree of rotation between 50% on the left and 50% on the right
        random_degree = np.random.uniform(-180, 180)

        # Rotate image without cutting sides off    
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

        (h,w,_) = rotated_img.shape
        rows = []
        for i in range(h):
            if (rotated_img[i, :, :] != 255).any():
                rows.append(rotated_img[i, :, :])
        
        rows = np.array(rows)
        
        cropped_img = []
        for j in range(w):
            if (rows[:, j, :] != 255).any():
                cropped_img.append(rows[:, j, :])
        
        return np.array(cropped_img)
    
    def random_resize(img):
        h, w, _ = img.shape
        random_size =  np.random.uniform(1.5, 3.)
        return cv2.resize(img, (int(w/random_size), int(h/random_size)))

    available_transformations = [random_rotation, random_resize]

    dataset_aug, labels_aug = copy.deepcopy(dataset), copy.deepcopy(labels)
    
    for transformation in available_transformations:
        for img, label in zip(dataset, labels):
            for _ in range(transformation_per_image):
                dataset_aug.append(transformation(img))
                labels_aug.append(label)
    
    return dataset_aug, labels_aug

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