import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List

# Return the dataframe given the filename
# Question 8 sub 1
def read_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

# Return the shape of the data
# Question 8 sub 2
def get_df_shape(df: pd.DataFrame) -> Tuple[int, int]:
    return df.shape

# Extract features "Lag1", "Lag2", and label "Direction"
# Question 8 sub 3
def extract_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[['Lag1', 'Lag2']]
    Y = df[['Direction']].squeeze()

    return X, Y

# Split the data into a train/test split
# Question 8 sub 4
def data_split(features: pd.DataFrame, label: pd.Series, test_size: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # split intro train and test
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=test_size)

    return x_train, y_train, x_test, y_test

# Write a function that returns score on test set with KNNs
# (use KNeighborsClassifier class)
# Question 8 sub 5
def knn_test_score(n_neighbors: int, x_train: np.ndarray, y_train: np.ndarray,
                   x_test: np.ndarray, y_test: np.ndarray) -> float:

    knnmodel = KNeighborsClassifier(n_neighbors=n_neighbors)
    knnmodel.fit(x_train, np.ravel(y_train, order='C'))

    # Predict using test data
    y_pred = knnmodel.predict(x_test)

    # Return accuracy score
    return metrics.accuracy_score(y_test, y_pred)


# Apply k-NN to a list of data
# You can use previously used functions (make sure they are correct)
# Question 8 sub 6
def knn_evaluate_with_neighbours(n_neighbors_min: int, n_neighbors_max: int,
                                 x_train: np.ndarray, y_train: np.ndarray,
                                 x_test: np.ndarray, y_test: np.ndarray
                                 ) -> List[float]:
    accuracies = []
    for k in range(n_neighbors_min, n_neighbors_max + 1):
        accuracies.append( knn_test_score(k, x_train, y_train, x_test, y_test) )

    return accuracies

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    df = read_data('Smarket.csv')

    # assert on df
    shape = get_df_shape(df)

    # assert on shape
    features, label = extract_features_label(df)
    x_train, y_train, x_test, y_test = data_split(features, label, 0.33)
    print(knn_test_score(1, x_train, y_train, x_test, y_test))

    acc = knn_evaluate_with_neighbours(1, 10, x_train, y_train, x_test, y_test)
    plt.plot(range(1, 11), acc)
    plt.show()