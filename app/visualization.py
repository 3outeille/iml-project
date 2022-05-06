import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def visualize_features(color_feature_extractor, img_train, img_test, n=10):
    histogram_train  = color_feature_extractor(img_train)
    histogram_test  = color_feature_extractor(img_test)
    dist_mat = cdist(histogram_test, histogram_train, metric='cosine')

    # Display the best matches for some bubbles
    max_res = 5
    interesting_bubble_ids = range(len(histogram_test))
    idx_of_best_matches_per_row = np.argsort(dist_mat, axis=1)

    for ii in interesting_bubble_ids[:n]:
        plt.figure(figsize=(12,8))
        columns = max_res + 1
        plt.subplot(1, columns, 1)
        plt.imshow(img_test[ii])
        plt.axis("off"); plt.title("Bubble %d"%(ii,))
        for jj in range(max_res):
            bb_idx = idx_of_best_matches_per_row[ii, jj]  # Read the id of each best match for current bubble
            plt.subplot(1, columns, jj+2)
            plt.imshow(img_train[bb_idx])
            plt.axis("off"); plt.title("b%d@%.3f" % (bb_idx, dist_mat[ii, bb_idx])) # display bubble id and dist.
        plt.show()

def get_labels_names(labels_association_path):
    labels_names = {}

    with open(labels_association_path, 'r') as file:

        for line in file:

            line = line.strip()
            if len(line) == 0:
                break

            label = int(line[:2])
            name = line[4:]
            labels_names[label] = name

    return labels_names