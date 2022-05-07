import pickle
import os
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC as SVM
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import dump_results


def train(classifer, feature_histogram, label_train):
    # TODO: Standard scaler?
    classifer.fit(feature_histogram, label_train)


def get_accuracy(prediction, label):
    accuracy = np.sum(np.array(prediction) ==
                      np.array(label)) / len(prediction)
    return accuracy


def evaluate(classifier, feature_histogram, label):
    prediction = classifier.predict(feature_histogram)
    accuracy = get_accuracy(prediction, label)
    return prediction, accuracy


def early_fusion(img_train, label_train, img_test, label_test, color_feature_extractor, shape_feature_extractor, n_neighbors, n_depth, ponderation):
    cwd = os.getcwd()
    # During developpement
    if cwd != "/app":
        cwd += "/app"

    def feature_extractor_fusion(color_feature_extractor, shape_feature_extractor):
        return lambda img: np.hstack((color_feature_extractor(img) * ponderation, shape_feature_extractor(img) * (1 - ponderation)))

    fused_feature_extractor = feature_extractor_fusion(
        color_feature_extractor, shape_feature_extractor)

    feature_train = fused_feature_extractor(img_train)
    feature_test = fused_feature_extractor(img_test)

    classifiers = {
        "knn_classifier": KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine'),
        "svm_classifier_linear": SVM(kernel='linear'),
        "svm_classifier_polynomial": SVM(kernel='poly'),
        "svm_classifier_rbf": SVM(kernel='rbf'),
        "random_forest_classifier": RandomForestClassifier(max_depth=n_depth),
        "cart": DecisionTreeClassifier()
    }

    best_accuracy = 0

    for name, clf in classifiers.items():
        print(name)
        print("\t\tTrain  ...")
        train(clf, feature_train, label_train)

        print("\t\tPrediction ...")
        prediction, accuracy = evaluate(
            clf, feature_test, label_test)

        print(f"\t\tAccuracy = {accuracy}\n")

        if accuracy > best_accuracy:
            print(f"Save current best accuracy ({name} classifier) ...")
            best_accuracy = accuracy
            # dump_results(prediction, label_test, f"app/{name}-result.csv")
            pickle.dump(clf, open(f"{cwd}/clf.pkl", "wb"))


def late_fusion(img_train, label_train, img_test, label_test, color_feature_extractor, shape_feature_extractor, n_neighbors, n_depth):
    cwd = os.getcwd()
    # During developpement
    if cwd != "/app":
        cwd += "/app"

    def soft_voting(color_predict_proba, shape_predict_proba, color_labels, shape_labels):
        def get_max_proba_indices(color_predict_proba, shape_predict_proba):
            classes_pred_proba = 0.6 * color_predict_proba + 0.4 * shape_predict_proba
            return np.argmax(classes_pred_proba, axis=1)

        max_proba_indces = get_max_proba_indices(
            color_predict_proba, shape_predict_proba)
        return max_proba_indces + 1

    classifiers = {
        "knn-knn": [KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine'), KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')],
        "knn-rand": [KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine'), RandomForestClassifier(max_depth=n_depth)],
        "knn-cart": [KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine'), DecisionTreeClassifier()],
        "random_forest-random_forest": [RandomForestClassifier(max_depth=n_depth), RandomForestClassifier(max_depth=n_depth)],
        "random_forest-knn": [RandomForestClassifier(max_depth=n_depth), KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')],
        "random_forest-cart": [RandomForestClassifier(max_depth=n_depth), DecisionTreeClassifier()],
        "cart-cart": [DecisionTreeClassifier(), DecisionTreeClassifier()],
        "cart-knn": [DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')],
        "cart-rand": [DecisionTreeClassifier(), RandomForestClassifier(max_depth=n_depth)]
    }

    color_feature_train = color_feature_extractor(img_train)
    shape_feature_train = shape_feature_extractor(img_train)

    color_feature_test = color_feature_extractor(img_test)
    shape_feature_test = shape_feature_extractor(img_test)

    best_accuracy = 0

    for name, clf in classifiers.items():
        print(f"Train {name} classifier on color features ...\n")
        train(clf[0], color_feature_train, label_train)
        print(f"Train {name} classifier on shape features ...\n")
        train(clf[1], shape_feature_train, label_train)

        color_labels = clf[0].predict(color_feature_test)
        shape_labels = clf[1].predict(shape_feature_test)
        prediction = soft_voting(clf[0].predict_proba(color_feature_test), clf[1].predict_proba(shape_feature_test), color_labels, shape_labels)

        accuracy = get_accuracy(prediction, label_test)
        print(f"late fusion {name} accuracy = {accuracy}\n")

        if accuracy > best_accuracy:
            print(f"Save current best accuracy ({name} classifier) ...")
            best_accuracy = accuracy
            # dump_results(prediction, label_test, f"app/{name}-result.csv")
            pickle.dump(clf, open(f"{cwd}/clf.pkl", "wb"))

        # dump_results(prediction, label_test, f"{name}-results")

def stacking_early_fusion(img_train, label_train, img_test, label_test, color_feature_extractor, shape_feature_extractor, random_seed, n_estimators, cv, ponderation):
    cwd = os.getcwd()
    # During developpement
    if cwd != "/app":
        cwd += "/app"

    def feature_extractor_fusion(color_feature_extractor, shape_feature_extractor):
        return lambda img: np.hstack((color_feature_extractor(img) * ponderation, shape_feature_extractor(img) * (1 - ponderation)))

    fused_feature_extractor = feature_extractor_fusion(
        color_feature_extractor, shape_feature_extractor)

    def base_model():
        level0 = list()
        level0.append(('cart', DecisionTreeClassifier()))
        level0.append(('rf', RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_seed)))

        model = StackingClassifier(
            estimators=level0, final_estimator=LogisticRegression(), cv=cv)
        return model
    
    def model_with_pipeline():
        level0 = list()
        level0.append(('cart', DecisionTreeClassifier()))
        level0.append(('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42))))
        model = StackingClassifier(estimators=level0, final_estimator=LogisticRegression(), cv=cv)
        return model

    feature_train = fused_feature_extractor(img_train)
    feature_test = fused_feature_extractor(img_test)

    stacking_clf = base_model()
    train(stacking_clf, feature_train, label_train)
    prediction1, accuracy1 = evaluate(
        stacking_clf, feature_test, label_test)
    print(f"stacking early clf accuracy = {accuracy1}\n")

    stacking_clf_with_pipeline = model_with_pipeline()
    train(stacking_clf_with_pipeline, feature_train, label_train)
    prediction2, accuracy2 = evaluate(
        stacking_clf_with_pipeline, feature_test, label_test)
    print(f"stacking early with pipeline clf accuracy = {accuracy2}\n")

    if accuracy1 >= accuracy2:
        pickle.dump(stacking_clf, open(f"{cwd}/clf.pkl", "wb"))
    else:
        pickle.dump(stacking_clf_with_pipeline, open(f"{cwd}/clf.pkl", "wb"))

def stacking_late_fusion(img_train, label_train, img_test, label_test, color_feature_extractor, shape_feature_extractor, random_seed, n_estimators, cv, ponderation):
    cwd = os.getcwd()
    # During developpement
    if cwd != "/app":
        cwd += "/app"

    def base_model():
        level0 = list()
        level0.append(('cart', DecisionTreeClassifier()))
        level0.append(('rf', RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_seed)))

        model = StackingClassifier(
            estimators=level0, final_estimator=LogisticRegression(), cv=cv)
        return model

    def soft_voting(color_predict_proba, shape_predict_proba, color_labels, shape_labels):
        def get_max_proba_indices(color_predict_proba, shape_predict_proba):
            classes_pred_proba = 0.6 * color_predict_proba + 0.4 * shape_predict_proba
            return np.argmax(classes_pred_proba, axis=1)

        max_proba_indces = get_max_proba_indices(
            color_predict_proba, shape_predict_proba)
        return max_proba_indces + 1

    color_feature_train = color_feature_extractor(img_train)
    shape_feature_train = shape_feature_extractor(img_train)

    color_feature_test = color_feature_extractor(img_test)
    shape_feature_test = shape_feature_extractor(img_test)

    color_stacking_clf = base_model()
    shape_stacking_clf = base_model()

    train(color_stacking_clf, color_feature_train, label_train)
    train(shape_stacking_clf, shape_feature_train, label_train)

    color_labels = color_stacking_clf.predict(color_feature_test)
    shape_labels = shape_stacking_clf.predict(shape_feature_test)

    prediction = soft_voting(color_stacking_clf.predict_proba(color_feature_extractor(
        img_test)), shape_stacking_clf.predict_proba(shape_feature_test), color_labels, shape_labels)
    accuracy = get_accuracy(prediction, label_test)

    print(f"stacking late clf accuracy = {accuracy}\n")

    pickle.dump(stacking_late_fusion, open(f"{cwd}/clf.pkl", "wb"))