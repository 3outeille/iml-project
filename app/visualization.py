from webbrowser import get
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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

def plot_confusion_matrix(labels_path, predictions, label_test):
    LINES = 10
    COLS = 6

    labels_names = get_labels_names(labels_path)
    f, axes = plt.subplots(LINES, COLS, figsize=(17, 10))
    axes = axes.ravel()

    for i in range(1, 58):
        y_true = label_test.copy()
        y_true[y_true != i] = 0

        y_pred = predictions.copy()
        y_pred[y_pred != i] = 0

        cm = confusion_matrix(y_true, y_pred, labels=[0, i])
        tn, fp, fn, tp = cm.ravel()

        cmap='autumn'
        if fp != 0 or fn != 0:
            cmap = 'winter'

        disp = ConfusionMatrixDisplay(cm,
                                    display_labels=['oth', i])
        disp.plot(ax=axes[i-1], values_format='.4g', cmap=cmap)
        disp.ax_.set_title(labels_names[i])

        if (i - 1) < (LINES - 1) * COLS:
            disp.ax_.set_xlabel('')

        if (i - 1) % COLS != 0:
            disp.ax_.set_ylabel('')
            
        disp.im_.colorbar.remove()

    f.tight_layout()
    plt.axis('off')
    plt.show()