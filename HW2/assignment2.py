################
################
## Q1
################
################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from typing import Tuple, List
from scipy.stats import sem, norm


# Download and read the data.
def read_train_data(filename: str) -> pd.DataFrame:
    '''
        read train data and return dataframe
    '''
    return pd.read_csv(filename)

def read_test_data(filename: str) -> pd.DataFrame:
    '''
        read train data and return dataframe
    '''
    return pd.read_csv(filename)

# Prepare your input data and labels
def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    '''
        Separate input data and labels, remove NaN values. Execute this for both dataframes.
        return tuple of numpy arrays(train_data, train_label, test_data, test_label).
    '''
    removeNanTrain = df_train.dropna()
    removeNanTest = df_test.dropna()

    x_train = removeNanTrain[["x"]].to_numpy()
    y_train = removeNanTrain[["y"]].to_numpy()

    x_test = removeNanTest[["x"]].to_numpy()
    y_test = removeNanTest[["y"]].to_numpy()

    return x_train, y_train, x_test, y_test


# Implement LinearRegression class
class LinearRegression:

    def __init__(self, learning_rate=0.00001, iterations=30):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coef_ = []

    # Function for model training
    def fit(self, X, Y):

        # weight initialization
        self.m = X.shape[0]  # Number of entries
        self.weight = np.zeros(X.shape[1])  # Number of features
        self.bias = 0

        self.X = X
        self.Y = Y

        # gradient descent learning
        for _ in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        y_pred = self.predict(self.X)
        intercept_gradient = -2 * np.sum(self.Y - y_pred) / self.m
        slope_gradient = -(2 * (self.X.T).dot(self.Y - y_pred)) / self.m

        self.bias = self.bias - (self.learning_rate * intercept_gradient)
        self.weight = self.weight - (self.learning_rate * slope_gradient)

        return self

    # Hypothetical function  h( x )
    def predict(self, X):
        return (self.weight * X) + self.bias


# Build your model
def build_model(train_X: np.array, train_y: np.array):
    '''
        Instantiate an object of LinearRegression class, train the model object
        using training data and return the model object
    '''
    lr = LinearRegression()
    lr.fit(train_X, train_y)
    return lr


# Make predictions with test set
def pred_func(model, X_test):
    '''
        return numpy array comprising of prediction on test set using the model
    '''
    return model.predict(X_test)


# Calculate and print the mean square error of your prediction
def MSE(y_test, pred):
    '''
        return the mean square error corresponding to your prediction
    '''
    return np.square(np.subtract(y_test, pred)).mean()


################
################
## Q2
################
################

# Download and read the data.
def read_training_data(filename: str) -> tuple:
    '''
        read train data into a dataframe df1, store the top 10 entries of the dataframe in df2
        and return a tuple of the form (df1, df2, shape of df1)   
    '''
    df1 = pd.read_csv(filename)
    df2 = df1.head(10)
    print(df2)
    return df1, df2, df1.shape


# Prepare your input data and labels
def data_clean(df_train: pd.DataFrame) -> tuple:
    '''
        check for any missing values in the data and store the missing values in series s, drop the entries corresponding 
        to the missing values and store dataframe in df_train and return a tuple in the form: (s, df_train)
    '''
    s = df_train.isnull()
    df_train = df_train.dropna(inplace=False)
    return s, df_train


def feature_extract(df_train: pd.DataFrame) -> tuple:
    '''
        New League is the label column.
        Separate the data from labels.
        return a tuple of the form: (features(dtype: pandas.core.frame.DataFrame), label(dtype: pandas.core.series.Series))
    '''
    x = df_train.iloc[:, :-1]
    y = df_train.iloc[:, -1:]
    return x, y


def data_preprocess(feature: pd.DataFrame) -> pd.DataFrame:
    '''
        Separate numerical columns from nonnumerical columns. (In pandas, check out .select dtypes(exclude = ['int64', 'float64']) and .select dtypes(
        include = ['int64', 'float64']). Afterwards, use get_dummies for transforming to categorical. Then concat both parts (pd.concat()).
        and return the concatenated dataframe.
    '''
    num_columns = feature.select_dtypes(include=['int64', 'float64'])
    nonnum_columns = feature.select_dtypes(exclude=['int64', 'float64'])
    to_categorical = pd.get_dummies(nonnum_columns, columns=["Division", "League"]).drop(columns=["Player"])
    return pd.concat([num_columns, to_categorical], axis=1)


def label_transform(labels: pd.Series) -> pd.Series:
    '''
        Transform the labels into numerical format and return the labels
    '''
    replace_policy = {"N": 1, "A": 0}
    labels = labels['NewLeague'].apply(lambda x: replace_policy[x])
    return labels


################
################
## Q3
################
################

# q31
def data_split(features: pd.DataFrame, label: pd.Series, random_state=42) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
        Split 80% of data as a training set and the remaining 20% of the data as testing set using the given random state
        return training and testing sets in the following order: X_train, X_test, y_train, y_test
    '''
    x_train, x_test, y_train, y_test = train_test_split(features, label, random_state=random_state, test_size=0.2)

    return x_train, x_test, y_train, y_test


# q32
def train_linear_regression(x_train: np.ndarray, y_train: np.ndarray):
    '''
        Instantiate an object of LinearRegression class, train the model object
        using training data and return the model object
    '''
    lr = LR()
    lr.fit(x_train, y_train)
    return lr


# q32
def train_logistic_regression(x_train: np.ndarray, y_train: np.ndarray, max_iter=1000000):
    '''
        Instantiate an object of LogisticRegression class, train the model object
        use provided max_iterations for training logistic model
        using training data and return the model object
    '''
    logR = LogisticRegression(max_iter=max_iter)
    logR.fit(x_train, y_train)
    return logR


# q33
def models_coefficients(linear_model, logistic_model) -> Tuple[np.ndarray, np.ndarray]:
    '''
        return the tuple consisting the coefficients for each feature for Linear Regression 
        and Logistic Regression Models respectively
    '''
    return linear_model.coef_, logistic_model.coef_


# q34 and q35
def linear_pred_and_area_under_curve(linear_model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[
    np.array, np.array, np.array, np.array, float]:
    '''
        return the tuple consisting the predictions and area under the curve measurements of Linear Regression 
        and Logistic Regression Models respectively in the following order 
        [linear_reg_pred, linear_reg_fpr, linear_reg_tpr, linear_threshold, linear_reg_area_under_curve]
        Finally plot the ROC Curve
    '''

    linear_reg_pred = linear_model.predict(x_test)
    linear_reg_fpr, linear_reg_tpr, linear_threshold = roc_curve(y_test, linear_reg_pred)
    linear_reg_area_under_curve = auc(linear_reg_fpr, linear_reg_tpr)

    return linear_reg_pred, linear_reg_fpr, linear_reg_tpr, linear_threshold, linear_reg_area_under_curve


# q34 and q35
def logistic_pred_and_area_under_curve(logistic_model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[
    np.array, np.array, np.array, np.array, float]:
    '''
        return the tuple consisting the predictions and area under the curve measurements of Linear Regression 
        and Logistic Regression Models respectively in the following order 
        [log_reg_pred, log_reg_fpr, log_reg_tpr, log_threshold, log_reg_area_under_curve]
        Finally plot the ROC Curve
    '''
    log_reg_pred = logistic_model.predict_proba(x_test)
    log_reg_fpr, log_reg_tpr, log_threshold = roc_curve(y_test, log_reg_pred[:,1])
    log_reg_area_under_curve = roc_auc_score(y_test, log_reg_pred[:,1])

    #print(auc(log_reg_fpr, log_reg_tpr))
    #print(roc_auc_score(y_test, logistic_model.predict_proba(x_test)[:,1]))

    return log_reg_pred[:,1], log_reg_fpr, log_reg_tpr, log_threshold, log_reg_area_under_curve


# q36
def optimal_thresholds(linear_threshold: np.ndarray, linear_reg_fpr: np.ndarray, linear_reg_tpr: np.ndarray,
                       log_threshold: np.ndarray, log_reg_fpr: np.ndarray, log_reg_tpr: np.ndarray) -> Tuple[
    float, float]:
    '''
        return the tuple consisting the thresholds of Linear Regression and Logistic Regression Models respectively

        Using Youden's J Statistic
    '''
    yd_linear = linear_reg_tpr - linear_reg_fpr
    optimal_i = np.argmax(yd_linear)
    optimal_threshold_linear = linear_threshold[optimal_i]

    yd_log = log_reg_tpr - log_reg_fpr
    optimal_i = np.argmax(yd_log)
    optimal_threshold_log = log_threshold[optimal_i]

    return optimal_threshold_linear, optimal_threshold_log


def stratified_k_fold_cross_validation(num_of_folds: int, shuffle: True, features: pd.DataFrame, label: pd.Series):
    '''
        split the data into 5 groups. Checkout StratifiedKFold in scikit-learn
    '''
    skf = StratifiedKFold(n_splits=num_of_folds, shuffle=shuffle)
    return skf


# q37
def train_test_folds(skf, num_of_folds: int, features: pd.DataFrame, labels: pd.Series) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, dict]:
    '''
        train and test in for loop with different training and test sets obatined from skf. 
        use a PENALTY of 12 for logistic regression model for training

        find features in each fold and store them in features_count array.

        populate auc_log and auc_linear arrays with roc_auc_score of each set trained on logistic regression and linear regression models respectively.

        populate f1_dict['log_reg'] and f1_dict['linear_reg'] arrays with f1_score of trained logistic and linear regression models on each set

        return features_count, auc_log, auc_linear, f1_dict dictionary
    '''

    features_count = np.zeros(num_of_folds)
    auc_log = np.zeros(num_of_folds)
    auc_linear = np.zeros(num_of_folds)
    f1_dict = {'log_reg': np.zeros(num_of_folds), 'linear_reg': np.zeros(num_of_folds)}
    skf = list(skf.split(features, labels))

    for i, (train_i, test_i) in enumerate(skf):
        x_train, x_test = features.iloc[train_i], features.iloc[test_i]
        y_train, y_test = labels.iloc[train_i], labels.iloc[test_i]

        logr = LogisticRegression(penalty='l2', max_iter=100000008)
        linr = LR()

        logr.fit(x_train, y_train)
        linr.fit(x_train, y_train)

        features_count[i] = x_train.shape[1]
        auc_log[i] = roc_auc_score(y_test, logr.predict(x_test))
        auc_linear[i] = roc_auc_score(y_test, linr.predict(x_test))
        f1_dict['log_reg'][i] = f1_score(y_test, logr.predict(x_test))
        f1_dict['linear_reg'][i] = f1_score(y_test, linr.predict(x_test).round().astype(int))

    return features_count, auc_log, auc_linear, f1_dict


# q38
def is_features_count_changed(features_count: np.array) -> bool:
    '''
       compare number of features in each fold (features_count array's each element)
       return true if features count doesn't change in each fold. else return false
    '''
    return len(set(features_count)) == 1  # turn to a set and check if the length is 1, aka all items are same


# q391 and q392
def mean_confidence_interval(data: np.array, confidence=0.95) -> Tuple[float, float, float]:
    '''
        To calculate mean and confidence interval, in scipy checkout .sem to find standard error of the mean of given data (AUROCs/ f1 scores of each model, linear and logistic trained on all sets). 
        Then compute Percent Point Function available in scipy and mutiply it with standard error calculated earlier to calculate h. 
        The required interval is from mean-h to mean+h
        return the tuple consisting of mean, mean -h, mean+h
    '''
    mean = np.mean(data)
    stand_err = sem(data)
    h = stand_err * norm.ppf((1 + confidence) / 2)

    lb = mean - h
    ub = mean + h

    return mean, lb, ub


if __name__ == "__main__":
    ################
    ################
    ## Q1
    ################
    ################
    data_path_train = "LinearRegression/train.csv"
    data_path_test = "LinearRegression/test.csv"
    df_train = read_train_data(data_path_train)
    df_test = read_test_data(data_path_test)

    train_X, train_y, test_X, test_y = prepare_data(df_train, df_test)
    model = build_model(train_X, train_y)

    # Make prediction with test set
    preds = pred_func(model, test_X)

    # Calculate and print the mean square error of your prediction
    mean_square_error = MSE(test_y, preds)
    print(mean_square_error)

    # plot your prediction and labels, you can save the plot and add in the report

    plt.plot(test_y, label='label')
    plt.plot(preds, label='pred')
    plt.legend()
    plt.show()

    ################
    ################
    ## Q2
    ################
    ################

    data_path_training = "Hitters.csv"

    train_df, df2, df_train_shape = read_training_data(data_path_training)
    s, df_train_mod = data_clean(train_df)
    features, label = feature_extract(df_train_mod)
    final_features = data_preprocess(features)
    final_label = label_transform(label)

    ################
    ################
    ## Q3
    ################
    ################

    num_of_folds = 5
    max_iter = 100000008
    X = final_features
    y = final_features
    auc_log = []
    auc_linear = []
    features_count = []
    f1_dict = {'log_reg': [], 'linear_reg': []}
    is_features_count_change = True

    X_train, X_test, y_train, y_test = data_split(final_features, final_label)

    linear_model = train_linear_regression(X_train, y_train)

    logistic_model = train_logistic_regression(X_train, y_train)

    linear_coef, logistic_coef = models_coefficients(linear_model, logistic_model)

    print(linear_coef)
    print(logistic_coef)

    linear_y_pred, linear_reg_fpr, linear_reg_tpr, linear_threshold, linear_reg_area_under_curve = linear_pred_and_area_under_curve(
        linear_model, X_test, y_test)

    log_y_pred, log_reg_fpr, log_reg_tpr, log_threshold, log_reg_area_under_curve = logistic_pred_and_area_under_curve(
        logistic_model, X_test, y_test)

    print("Linear AOC: ", linear_reg_area_under_curve)
    print("Logisitic AOC: ", log_reg_area_under_curve)

    plt.plot(log_reg_fpr, log_reg_tpr, label='logistic')
    plt.plot(linear_reg_fpr, linear_reg_tpr, label='linear')
    plt.legend()
    plt.show()

    linear_optimal_threshold, log_optimal_threshold = optimal_thresholds(linear_threshold, linear_reg_fpr,
                                                                         linear_reg_tpr, log_threshold, log_reg_fpr,
                                                                         log_reg_tpr)
    print("Linear Opt. Threshold: ", linear_optimal_threshold)
    print("Logisitic Opt. Threshold: ", log_optimal_threshold)

    skf = stratified_k_fold_cross_validation(num_of_folds, True, final_features, final_label)
    features_count, auc_log, auc_linear, f1_dict = train_test_folds(skf, num_of_folds, final_features, final_label)

    print("Does features change in each fold?")

    # call is_features_count_changed function and return true if features count changes in each fold. else return false
    is_features_count_change = is_features_count_changed(features_count)

    print(is_features_count_change)

    auc_linear_mean, auc_linear_open_interval, auc_linear_close_interval = 0, 0, 0
    auc_log_mean, auc_log_open_interval, auc_log_close_interval = 0, 0, 0

    f1_linear_mean, f1_linear_open_interval, f1_linear_close_interval = 0, 0, 0
    f1_log_mean, f1_log_open_interval, f1_log_close_interval = 0, 0, 0

    # Find mean and 95% confidence interval for the AUROCs for each model and populate the above variables accordingly
    # Hint: use mean_confidence_interval function and pass roc_auc_scores of each fold for both models (ex: auc_log)
    # Find mean and 95% confidence interval for the f1 score for each model.

    auc_linear_mean, auc_linear_open_interval, auc_linear_close_interval = mean_confidence_interval(
        auc_linear)
    auc_log_mean, auc_log_open_interval, auc_log_close_interval = mean_confidence_interval(auc_log)

    f1_linear_mean, f1_linear_open_interval, f1_linear_close_intervel = mean_confidence_interval(
        f1_dict['linear_reg'])
    f1_log_mean, f1_log_open_interval, f1_log_close_interval = mean_confidence_interval(f1_dict['log_reg'])


    print(auc_linear_mean, auc_linear_open_interval, auc_linear_close_interval)
    print(auc_log_mean, auc_log_open_interval, auc_log_close_interval)

    print(f1_linear_mean, f1_linear_open_interval, f1_linear_close_intervel)
    print(f1_log_mean, f1_log_open_interval, f1_log_close_interval)








