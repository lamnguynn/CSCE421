#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List, Optional, Any, Callable, Dict, Union
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.utils import resample
import random
from typeguard import typechecked

random.seed(42)
np.random.seed(42)


@typechecked
def read_data(filename: str) -> pd.DataFrame:
    """
    Read the data from the filename. Load the data it in a dataframe and return it.
    """
    return pd.read_csv(filename)


@typechecked
def data_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Follow all the preprocessing steps mentioned in Problem 2 of HW2 (Problem 2: Coding: Preprocessing the Data.)
    Return the final features and final label in same order
    You may use the same code you submiited for problem 2 of HW2
    """

    #Clean up the data
    df = df.dropna(inplace=False)

    #Feature extraction
    feature = df.iloc[:, :-1]
    labels = df.iloc[:, -1:]

    #Process the features
    num_columns = feature.select_dtypes(include=['int64', 'float64'])
    nonnum_columns = feature.select_dtypes(exclude=['int64', 'float64'])
    to_categorical = pd.get_dummies(nonnum_columns, columns=["Division", "League"]).drop(columns=["Player"])
    feature = pd.concat([num_columns, to_categorical], axis=1)

    #Process the labels
    replace_policy = {"N": 1, "A": 0}
    labels = labels['NewLeague'].apply(lambda x: replace_policy[x])

    return feature, labels


@typechecked
def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split 80% of data as a training set and the remaining 20% of the data as testing set
    return training and testing sets in the following order: X_train, X_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=test_size)

    return x_train, x_test, y_train, y_test


@typechecked
def train_ridge_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter: int = int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Ridge Regression, train the model object using training data for the given N-bootstraps
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary

    Rest of the provided handles the return part
    """
    n_bootstraps = int(1e3)
    aucs = {"ridge": []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    for i in range(n_bootstraps):
        for lambda_val in lambda_vals:
            ridge = Ridge(alpha=lambda_val, max_iter=max_iter)
            ridge.fit(x_train, y_train)
            y_pred = ridge.predict(x_test)
            ridge_auc = roc_auc_score(y_test, y_pred)
            aucs['ridge'].append(ridge_auc)

    grouped = {}
    for i in range(len(lambda_vals)):
        grouped[lambda_vals[i]] = []
    for i in range(len(aucs['ridge'])):
        grouped[lambda_vals[i % len(lambda_vals)]].append(aucs['ridge'][i])

    print("ridge mean AUCs:")
    ridge_mean_auc = {}
    ridge_aucs = pd.DataFrame(grouped)
    for lambda_val, ridge_auc in zip(lambda_vals, ridge_aucs.mean()):
        ridge_mean_auc[lambda_val] = ridge_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % ridge_auc)
    return ridge_mean_auc


@typechecked
def train_lasso(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter=int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Lasso Model, train the object using training data for the given N-bootstraps
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary

    Rest of the provided handles the return part
    """
    n_bootstraps = int(1e3)
    aucs = {"lasso": []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    for i in range(n_bootstraps):
        for lambda_val in lambda_vals:
            lasso = Lasso(alpha=lambda_val, max_iter=max_iter)
            lasso.fit(x_train, y_train)
            y_pred = lasso.predict(x_test)
            lasso_auc = roc_auc_score(y_test, y_pred)
            aucs['lasso'].append(lasso_auc)

    grouped = {}
    for i in range(len(lambda_vals)):
        grouped[lambda_vals[i]] = []
    for i in range(len(aucs['lasso'])):
        grouped[lambda_vals[i % len(lambda_vals)]].append(aucs['lasso'][i])

    print("lasso mean AUCs:")
    lasso_mean_auc = {}
    lasso_aucs = pd.DataFrame(grouped)
    for lambda_val, lasso_auc in zip(lambda_vals, lasso_aucs.mean()):
        lasso_mean_auc[lambda_val] = lasso_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % lasso_auc)

    print(lasso_mean_auc)
    return lasso_mean_auc


@typechecked
def ridge_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Ridge, np.ndarray]:
    """
    return the tuple consisting of trained Ridge model with alpha as optimal_alpha and the coefficients
    of the model
    """
    ridge = Ridge(max_iter=max_iter, alpha=optimal_alpha)
    ridge.fit(x_train, y_train)
    return ridge, ridge.coef_

@typechecked
def lasso_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Lasso, np.ndarray]:
    """
    return the tuple consisting of trained Lasso model with alpha as optimal_alpha and the coefficients
    of the model
    """
    lasso = Lasso(max_iter=max_iter, alpha=optimal_alpha)
    lasso.fit(x_train, y_train)
    return lasso, lasso.coef_


@typechecked
def ridge_area_under_curve(
    model_R, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    return area under the curve measurements of trained Ridge model used to find coefficients,
    i.e., model tarined with optimal_aplha
    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    r_pred = model_R.predict(x_test)
    r_fpr, r_tpr, _ = roc_curve(y_test, r_pred)
    r_area_under_curve = auc(r_fpr, r_tpr)

    '''
    plt.figure()
    plt.plot(r_fpr, r_tpr)
    plt.title('Reciever Operating Characteristic - Ridge Regression Model')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    '''

    return r_area_under_curve

@typechecked
def lasso_area_under_curve(
    model_L, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    return area under the curve measurements of Lasso Model,
    i.e., model tarined with optimal_aplha
    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    l_pred = model_L.predict(x_test)
    l_fpr, l_tpr, _ = roc_curve(y_test, l_pred)
    l_area_under_curve = auc(l_fpr, l_tpr)

    '''
    plt.figure()
    plt.plot(l_fpr, l_tpr)
    plt.title('Reciever Operating Characteristic - Lasso Regression Model')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    '''

    return l_area_under_curve


class Node:
    @typechecked
    def __init__(
        self,
        split_val: float,
        data: Any = None,
        left: Any = None,
        right: Any = None,
        index: Any = None
    ) -> None:
        if left is not None:
            assert isinstance(left, Node)

        if right is not None:
            assert isinstance(right, Node)

        self.left = left
        self.right = right
        self.split_val = split_val  # value (of a variable) on which to split. For leaf nodes this is label/output value
        self.data = data  # data can be anything! we recommend dictionary with all variables you need
        self.index = index

class TreeRegressor:
    @typechecked
    def __init__(self, data: np.ndarray, max_depth: int) -> None:
        self.data = (
            data  # last element of each row in data is the target variable
        )
        self.max_depth = max_depth  # maximum depth

    @typechecked
    def build_tree(self) -> Node:
        """
        Build the tree
        """
        root = self.get_best_split(data=self.data)
        self.split(node=root, depth=1)

        return root

    @typechecked
    def mean_squared_error(
        self, left_split: np.ndarray, right_split: np.ndarray
    ) -> float:
        """
        Calculate the mean squared error for a split dataset
        left split is a list of rows of a df, rightmost element is label
        return the sum of mse of left split and right split
        """
        # Guard statement
        if len(left_split) == 0 or len(right_split) == 0:
            return 0

        left_mean = np.mean(left_split[:, -1])
        right_mean = np.mean(right_split[:, -1])

        mse = np.mean((left_split[:, -1] - left_mean) ** 2) + np.mean((right_split[:, -1] - right_mean)**2)
        return mse

    @typechecked
    def one_step_split(
        self, index: int, value: float, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a dataset based on an attribute and an attribute value
        index is the variable to be split on (left split < threshold)
        returns the left and right split each as list
        each list has elements as `rows' of the df
        """
        left = data[data[:, index] < value]
        right = data[data[:, index] >= value]

        return left, right

    @typechecked
    def split(self, node: Node, depth: int) -> None:
        """
        Do the split operation recursively
        """

        #Base case
        if node.left is None or node.right is None:
            return
        if depth >= self.max_depth:
            node.left = Node(np.mean(node.left.data[:, -1]))
            node.right = Node(np.mean(node.right.data[:, -1]))
            return

        node.left = self.get_best_split(node.left.data)
        self.split(node.left, depth + 1)

        node.right = self.get_best_split(node.right.data)
        self.split(node.right, depth + 1)

    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset AND create a Node
        """

        m, n = data.shape
        best_score = float("inf")
        best_value = None
        best_index = None

        for split_ind in range(n - 1):
            for value in data[:, split_ind]:
                left_split, right_split = self.one_step_split(split_ind, value, data)

                if len(left_split) == 0 or len(right_split) == 0:
                    continue

                score = self.mean_squared_error(left_split, right_split)

                if score < best_score:
                    best_score = score
                    best_value = value
                    best_index = split_ind

        if best_score < self.mean_squared_error(data, data):
            left_split, right_split = self.one_step_split(best_index, best_value, data)

            return Node(split_val=best_value,
                        data=data,
                        index=best_index,
                        left=Node(split_val=np.mean(left_split[:, -1]), data=left_split),
                        right=Node(split_val=np.mean(right_split[:, -1]), data=right_split)
                        )

        return Node(split_val=np.mean(data[:, -1]), data=data)

@typechecked
def compare_node_with_threshold(node: Node, row: np.ndarray) -> bool:
    """
    Return True if node's value > row's value (of the variable)
    Else False
    """
    return bool(node.split_val > row[node.index])


@typechecked
def predict(
    node: Node, row: np.ndarray, comparator: Callable[[Node, np.ndarray], bool]
) -> float:
    if node.left is None or node.right is None:
        return node.split_val

    if comparator(node, row):
        return predict(node.left, row, comparator)
    else:
        return predict(node.right, row, comparator)


class TreeClassifier(TreeRegressor):
    def build_tree(self):
        ## Note: You can remove this if you want to use build tree from Tree Regressor
        root = self.get_best_split(self.data)
        self.split(root, 1)

        return root

    @typechecked
    def gini_index(
        self,
        left_split: np.ndarray,
        right_split: np.ndarray,
        classes: List[float],
    ) -> float:
        """
        Calculate the Gini index for a split dataset
        Similar to MSE but Gini index instead

        Acknowledges: https://www.learnbymarketing.com/481/decision-tree-flavors-gini-info-gain/
        """
        gini = 0
        num_of_instances = len(left_split) + len(right_split)
        for split in (left_split, right_split):
            # Check for a valid split
            if len(split) == 0:
                continue

            # Calculating percent branch
            score = 0
            for clas in classes:
                p_i = np.sum(split[:, -1] == clas) / len(split)
                score += (p_i ** 2)

            gini += (1 - score) * (len(split) / num_of_instances)

        return gini

    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset
        """

        classes = list(set(row[-1] for row in data))

        if len(classes) == 1:
            return Node(split_val=classes[0], data=data)

        m, n = data.shape
        best_score = float("inf")
        best_value = None
        best_index = None

        for split_ind in range(n - 1):
            for value in np.unique(data[:, split_ind]):
                left_split, right_split = self.one_step_split(split_ind, value, data)

                if len(left_split) == 0 or len(right_split) == 0:
                    continue

                score = self.gini_index(left_split, right_split, classes)

                if score < best_score:
                    best_score = score
                    best_value = value
                    best_index = split_ind

        if best_score < self.gini_index(data, data, classes):
            left_split, right_split = self.one_step_split(best_index, best_value, data)
            return Node(split_val=best_value,
                        data=data,
                        index=best_index,
                        left=self.get_best_split(right_split),
                        right=self.get_best_split(left_split))
        return Node(split_val=classes[0], data=data)


if __name__ == "__main__":
    '''
    # Question 1
    filename = ""  # Provide the path of the dataset
    df = read_data(filename)
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    max_iter = 1e8
    final_features, final_label = data_preprocess(df)
    x_train, x_test, y_train, y_test = data_split(
        final_features, final_label, 0.2
    )
    ridge_mean_acu = train_ridge_regression(x_train, y_train, x_test, y_test)
    lasso_mean_acu = train_lasso(x_train, y_train, x_test, y_test)
    model_R, ridge_coeff = ridge_coefficients(x_train, y_train, 10)
    model_L, lasso_coeff = lasso_coefficients(x_train, y_train, 0.1)
    ridge_auc = ridge_area_under_curve(model_R, x_test, y_test)
    # Plot the ROC curve of the Ridge Model. Include axes labels,
    # legend and title in the Plot. Any of the missing
    # items in plot will result in loss of points.

    lasso_auc = lasso_area_under_curve(model_L, x_test, y_test)
    # Plot the ROC curve of the Lasso Model.
    # Include axes labels, legend and title in the Plot.
    # Any of the missing items in plot will result in loss of points.
    '''

    # SUB Q1
    csvname = "noisy_sin_subsample_2.csv"
    data_regress = np.loadtxt(csvname, delimiter=",")
    data_regress = np.array([[x, y] for x, y in zip(*data_regress)])
    plt.figure()
    plt.scatter(data_regress[:, 0], data_regress[:, 1])
    plt.xlabel("Features, x")
    plt.ylabel("Target values, y")
    plt.show()
    mse_depths = []

    for depth in range(1, 5):
        regressor = TreeRegressor(data_regress, depth)
        tree = regressor.build_tree()
        mse = 0.0
        for data_point in data_regress:
            mse += (
                data_point[1]
                - predict(tree, data_point, compare_node_with_threshold)
            ) ** 2
        mse_depths.append(mse / len(data_regress))
    plt.figure()
    plt.plot(mse_depths)
    plt.xlabel("Depth")
    plt.ylabel("MSE")
    plt.show()
    # SUB Q2
    csvname = "new_circle_data.csv"
    data_class = np.loadtxt(csvname, delimiter=",")
    data_class = np.array([[x1, x2, y] for x1, x2, y in zip(*data_class)])
    plt.figure()
    plt.scatter(
        data_class[:, 0], data_class[:, 1], c=-data_class[:, 2], cmap="bwr"
    )
    plt.xlabel("Features, x1")
    plt.ylabel("Features, x2")
    plt.show()
    accuracy_depths = []
    for depth in range(1, 8):
        classifier = TreeClassifier(data_class, depth)
        tree = classifier.build_tree()
        correct = 0.0
        for data_point in data_class:
            correct += float(
                data_point[2]
                == predict(tree, data_point, compare_node_with_threshold)
            )
        accuracy_depths.append(correct / len(data_class))
    # Plot the MSE
    plt.figure()
    plt.plot(accuracy_depths)
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.show()