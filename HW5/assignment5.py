#!/usr/bin/env python

print("Acknowledgment:")
print("https://github.com/pritishuplavikar/Face-Recognition-on-Yale-Face-Dataset")

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import glob
from numpy import linalg as la
import random
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import os

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import sklearn
from typing import Tuple, List
from typeguard import typechecked
from sklearn.preprocessing import MinMaxScaler


@typechecked
def qa1_load(folder_path:str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the dataset (tuple of x, y the label).

    x should be of shape [165, 243 * 320]
    label can be extracted from the subject number in filename. ('subject01' -> '01 as label)
    """
    features = []
    labels= []

    # loop through all the files in the folder
    for filename in os.listdir(folder_path):
        if "subject" not in filename:
            continue
        features.append(mpimg.imread(os.path.join(folder_path, filename)).flatten()) #flatten to get the dimensions right
        labels.append(filename[7:9])

    return np.array(features), np.array(labels)


@typechecked
def qa2_preprocess(dataset:np.ndarray) -> np.ndarray:
    """
    returns data (x) after pre processing

    hint: consider using preprocessing.MinMaxScaler

    Acknowledgement: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """

    scaler = MinMaxScaler()
    return scaler.fit_transform(dataset)


@typechecked
def qa3_calc_eig_val_vec(dataset:np.ndarray, k:int)-> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    Calculate eig values and eig vectors.
    Use PCA as imported in the code to create an instance
    return them as tuple PCA, eigen_value, eigen_vector

    Acknowledgements: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
                     https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-vectors-from-sklearn-pca
    """
    pca = PCA(n_components=k)
    pca.fit(dataset)

    #for readability
    eigen_value = pca.explained_variance_
    eigen_vector = pca.components_

    return pca, eigen_value, eigen_vector


def qb_plot_written(eig_values:np.ndarray):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots
    """

    # Find how many components do we need to capture 50% of the energy
    cumsum = np.cumsum(eig_values) / np.sum(eig_values)
    num_comp = np.argmax(cumsum >= 0.5) + 1 # need to compensate cuz list indexing
    print(eig_values)
    # Plot displaying first k eigenvalues
    k = 10
    plt.plot(np.arange(1, k + 1), cumsum[:k])
    plt.xlabel("Number of components (k)")
    plt.ylabel("Energy")
    plt.title("First k eigenvalues plot")
    plt.show()


@typechecked
def qc1_reshape_images(pca:PCA, dim_x = 243, dim_y = 320) -> np.ndarray:
    """
    reshape the pca components into the shape of original image so that it can be visualized
    """
    eigen_vector = pca.components_
    eigen_faces = eigen_vector.reshape((-1, dim_x, dim_y))

    return eigen_faces


def qc2_plot(org_dim_eig_faces:np.ndarray):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots
    """

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axs.flatten()):
        # Pick a random eigen face
        rand_index = random.randint(0,len(org_dim_eig_faces)-1)
        ax.imshow(org_dim_eig_faces[rand_index], cmap='gray')

        # Remove the x and y labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Show title
        ax.set_title(f"Eigenface {i + 1}")
    plt.show()


@typechecked
def qd1_project(dataset:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the projection of the dataset 
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    return pca.transform(dataset)


@typechecked
def qd2_reconstruct(projected_input:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the reconstructed image given the pca components
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    return pca.inverse_transform(projected_input)


def qd3_visualize(dataset:np.ndarray, pca:PCA, dim_x = 243, dim_y = 320):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots. You can use other functions that you coded up for the assignment
    """
    orig_imgs = [random.choice(dataset), random.choice(dataset)]
    num_comps = [1, 3, 10, 20, 30, 40, 50, 100]
    fig, axes = plt.subplots(ncols=2, nrows=len(num_comps) + 1)

    for i, im in enumerate(orig_imgs):
        for j, k in enumerate(num_comps):
            pca, _, _ = qa3_calc_eig_val_vec(dataset, k)
            trans_img = qd1_project(orig_imgs[i].reshape(1, -1), pca)
            recon_img = qd2_reconstruct(trans_img, pca)

            axes[0][i].imshow(im.reshape(dim_x, dim_y), cmap='gray')
            axes[0][i].set_title('OG', rotation='vertical', x=-0.1,y=0)
            axes[j + 1][i].imshow(recon_img.reshape(dim_x, dim_y), cmap='gray')
            axes[j + 1][i].set_title(f"{k}", rotation='vertical', x=-0.1,y=0)

            axes[0][i].set_xticks([])
            axes[0][i].set_yticks([])
            axes[j + 1][i].set_xticks([])
            axes[j + 1][i].set_yticks([])

    plt.show()

@typechecked
def qe1_svm(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold).

    Hint: you can pick 5 `k' values uniformly distributed

    Acknowledgement: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
                     https://stackoverflow.com/questions/42245617/combining-principal-component-analysis-and-support-vector-machine-in-a-pipeline
    """
    trainX_transf = pca.transform(trainX)
    num_components_range = [10, 30, 50, 70, 90]

    best_k = None
    best_accuracy = 0.0

    for k in num_components_range:
        avg_accuracy = 0.0
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in kfold.split(trainX_transf):
            inner_trainX, inner_testX = trainX_transf[train_index], trainX_transf[test_index]
            inner_trainY, inner_testY = trainY[train_index], trainY[test_index]

            svm = SVC(kernel='rbf')
            svm.fit(inner_trainX[:, :k], inner_trainY)

            accuracy = accuracy_score(inner_testY, svm.predict(inner_testX[:, :k]))
            avg_accuracy += accuracy / kfold.n_splits

        if avg_accuracy > best_accuracy:
            best_k = k
            best_accuracy = avg_accuracy

    svm = SVC(kernel='rbf')
    svm.fit(trainX_transf[:, :best_k], trainY)
    test_accuracy = accuracy_score(trainY, svm.predict(trainX_transf[:, :best_k]))

    print(best_k, test_accuracy)
    return best_k, test_accuracy

@typechecked
def qe2_lasso(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold) in that order.

    Hint: you can pick 5 `k' values uniformly distributed
    """
    trainY = trainY.astype(np.float64)
    trainX_transf = pca.transform(trainX)
    num_components_range = [10, 30, 50, 70, 90]

    best_k = None
    best_accuracy = 0

    for k in num_components_range:
        avg_accuracy = 0.0
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in kfold.split(trainX_transf):
            inner_trainX, inner_testX = trainX_transf[train_index], trainX_transf[test_index]
            inner_trainY, inner_testY = trainY[train_index], trainY[test_index]

            lasso = Lasso(alpha=0.01)
            lasso.fit(inner_trainX[:, :k], inner_trainY)

            accuracy = 1 - accuracy_score(inner_testY, lasso.predict(inner_testX[:, :k]).round())
            avg_accuracy += accuracy / kfold.n_splits

        if avg_accuracy > best_accuracy:
            best_k = k
            best_accuracy = avg_accuracy

    lasso = Lasso(alpha=0.01)
    lasso.fit(trainX_transf[:, :best_k], trainY)
    test_accuracy = 1 - accuracy_score(trainY, lasso.predict(trainX_transf[:, :best_k]).round())

    return best_k, test_accuracy


if __name__ == "__main__":

    faces, y_target = qa1_load("./archive")
    dataset = qa2_preprocess(faces)
    pca, eig_values, eig_vectors = qa3_calc_eig_val_vec(dataset, len(dataset))

    qb_plot_written(eig_values)

    num = len(dataset)
    org_dim_eig_faces = qc1_reshape_images(pca)
    #qc2_plot(org_dim_eig_faces)

    #qd3_visualize(dataset, pca)
    #best_k, result = qe1_svm(dataset, y_target, pca)
    #print(best_k, result)
    best_k, result = qe2_lasso(dataset, y_target, pca)
    print(best_k, result)
