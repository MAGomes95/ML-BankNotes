
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import math as mt
from mlxtend.evaluate import mcnemar_table

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns',10)
import NaiveBayesClassifier



# Logistic Regression ----------

def err_calc_lr(X, y, train_int, valid_int, C=1.0):
    Log_Reg = LogisticRegression(C=C)
    Log_Reg.fit(X[train_int, :], y[train_int])

    train_score = 1 - Log_Reg.score(X[train_int, :], y[train_int])
    validation_score = 1 - Log_Reg.score(X[valid_int, :], y[valid_int])

    return train_score, validation_score


def train_C(X, y):
    C_value = [1.0]
    kf = StratifiedKFold(n_splits=5)
    t_error = list()
    cv_error = list()

    for iteration in range(20):
        train_error, validation_error = 0, 0
        for train_inc, valid_inc in kf.split(y, y):
            train_error += err_calc_lr(X, y, train_inc, valid_inc, C=C_value[iteration])[0]
            validation_error += err_calc_lr(X, y, train_inc, valid_inc, C=C_value[iteration])[1]
        C_value.append(C_value[iteration] * 2)
        t_error.append(train_error)
        cv_error.append(validation_error)

    C_value = [mt.log10(x) for x in C_value]

    plt.title('Plot of Training and Validation Errors for Logistic Regression')
    plt.plot(C_value[:-1], t_error, '-', label='Training Error')
    plt.plot(C_value[:-1], cv_error, '-', label='Validation Error')
    plt.xlabel('Log10(C)')
    plt.ylabel('Error Value')
    plt.legend()
    plt.show()

    ymin = min(cv_error)
    xpos = cv_error.index(ymin)
    xmin = C_value[xpos]

    best_C = 10 ** (xmin)
    print("The best C for regularization is: {}".format(best_C))
    return best_C


# K Nearest Neigbours --------- 

def err_calc_knn(X, y, train_int, valid_int, k_value):
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(X[train_int, :], y[train_int])

    train_score = 1 - knn_model.score(X[train_int, :], y[train_int])
    validation_score = 1 - knn_model.score(X[valid_int, :], y[valid_int])

    return train_score, validation_score


def train_k(X, y):
    k_values = [j for j in range(1, 40, 2)]
    kf = StratifiedKFold(n_splits=5)
    t_error = list()
    cv_error = list()

    for k_value in k_values:
        train_error, validation_error = 0, 0
        for train_inc, valid_inc in kf.split(y, y):
            train_error += err_calc_knn(X, y, train_inc, valid_inc, k_value)[0]
            validation_error += err_calc_knn(X, y, train_inc, valid_inc, k_value)[1]
        t_error.append(train_error)
        cv_error.append(validation_error)

    plt.title('Plot of Training and Validation Errors for KNN')
    plt.plot(k_values, t_error, '-', label='Training Error')
    plt.plot(k_values, cv_error, '-', label='Validation Error')
    plt.xlabel('K')
    plt.ylabel('Error Value')
    plt.legend()
    plt.show()

    ymin = min(cv_error)
    xpos = cv_error.index(ymin)
    xmin = k_values[xpos]
    best_k = xmin

    print("The best K for KNN is: {}".format(best_k))
    return best_k


# Naive Bayes with KDE ----------

def err_calc_nb(X, y, train_int, valid_int, bandwidth_value):
    nb_model = NaiveBayesClassifier.NaiveBayesClassifier(band_value=bandwidth_value)
    nb_model.fit(X[train_int, :], y[train_int])

    y_pred_train = nb_model.predict(X[train_int, :])
    y_pred_validation = nb_model.predict(X[valid_int, :])

    train_score = 1 - nb_model.score(y[train_int], y_pred_train)
    validation_score = 1 - nb_model.score(y[valid_int], y_pred_validation)

    return train_score, validation_score


def train_bd(X, y):
    band_values = np.arange(0.01, 1, 0.02)
    kf = StratifiedKFold(n_splits=5)
    t_error = list()
    cv_error = list()

    for band_value in band_values:
        train_error, validation_error = 0, 0
        for train_inc, valid_inc in kf.split(y, y):
            train_error += err_calc_nb(X, y, train_inc, valid_inc, band_value)[0]
            validation_error += err_calc_nb(X, y, train_inc, valid_inc, band_value)[1]
        t_error.append(train_error)
        cv_error.append(validation_error)

    plt.title('Plot of Training and Validation Errors for Naive Bayes')
    plt.plot(band_values, t_error, '-', label='Training Error')
    plt.plot(band_values, cv_error, '-', label='Validation Error')
    plt.xlabel('Band Width')
    plt.ylabel('Error Value')
    plt.legend()
    plt.show()

    ymin = min(cv_error)
    xpos = cv_error.index(ymin)
    xmin = band_values[xpos]

    best_band = xmin
    print("The best Band Width for the Kernel Density is: {}".format(best_band))

    return best_band

#Score Summary ----------

def summarize_scores(X,y,test_x,test_y):

    clfs = {'LR': LogisticRegression(C=best_C),
            'KNN':KNeighborsClassifier(n_neighbors=best_k),
            'Naive Bayes Classifier': NaiveBayesClassifier.NaiveBayesClassifier(band_value=best_bd)}

    df = pd.DataFrame(data={"ROC AUC": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-score": [], "CF": [], "y_pred": []})


    for key, clf in clfs.items():

        clf.fit(X, y)
        y_pred_test = clf.predict(test_x)

        roc_auc = roc_auc_score(test_y, y_pred_test)
        accuracy = accuracy_score(test_y, y_pred_test)
        precision = precision_score(test_y, y_pred_test)
        recall = recall_score(test_y, y_pred_test)
        f1_scores = f1_score(test_y, y_pred_test)
        confusion_matrixs = confusion_matrix(test_y, y_pred_test)

        df = df.append({'Model': key, 'ROC AUC': roc_auc, "Accuracy": accuracy, "Precision": precision,
                        "Recall": recall, "F1-score": f1_scores, "CF": confusion_matrixs,
                        "y_pred": y_pred_test}, ignore_index=True)

    y_pred = df.pop("y_pred")
    cm = df.pop("CF")
    return df.set_index("Model", drop=True), y_pred, cm


#McNemar test ----------

def nemarTest(e01, e10):
    return ((abs(e01 - e10) - 1) ** 2) / (e01 + e10)


def summarize_nemar(confusion_matrix, y_pred):
    errorsLog = confusion_matrix[0][1, 0] + confusion_matrix[0][0, 1]
    errorsKnn = confusion_matrix[1][1, 0] + confusion_matrix[1][0, 1]
    errorsBayse = confusion_matrix[2][1, 0] + confusion_matrix[2][0, 1]

    logVsKnn = mcnemar_table(y_target=y_test,
                             y_model1=y_pred[0],
                             y_model2=y_pred[1])


    logVsBayse = mcnemar_table(y_target=y_test,
                               y_model1=y_pred[0],
                               y_model2=y_pred[2])


    BayseVsKnn = mcnemar_table(y_target=y_test,
                               y_model1=y_pred[2],
                               y_model2=y_pred[1])


    df = pd.DataFrame(data={"Models vs Models": [], "McNemar's Test": [], "Model 1 Error": [], "Model 2 Error": []})
    df = df.append({"Models vs Models": "LR vs KNN",
                    "McNemar's Test": nemarTest(logVsKnn[1, 0], logVsKnn[0, 1]),
                    'Model 1 Error': errorsLog, "Model 2 Error": errorsKnn}, ignore_index=True)
    df = df.append({"Models vs Models": "LR vs Naive Bayes",
                    "McNemar's Test":    nemarTest(logVsBayse[1, 0], logVsBayse[0, 1]),
                    'Model 1 Error': errorsLog, "Model 2 Error": errorsBayse}, ignore_index=True)
    df = df.append({"Models vs Models": "Naive Bayes vs KNN",
                    "McNemar's Test":     nemarTest(BayseVsKnn[1, 0], BayseVsKnn[0, 1]),
                    'Model 1 Error': errorsBayse, "Model 2 Error": errorsKnn}, ignore_index=True)

    return df.set_index("Models vs Models", drop=True)


#==============================================================================================================#

#Main()
data = pd.read_csv("TP1-data.csv",
                   names=['Wavelet_variance', 'Wavelet_skewness', 'Wavelet_curtosis', 'Image_entropy', 'target'])

#Put it in a numpy!
data = data.values

#Scalling the Data
Scaler = StandardScaler()

data[:, :4] = Scaler.fit_transform(data[:, :4])

X, y = data[:, :4], data[:, 4]

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 / 3), random_state=3, stratify=y)

#Get the best parameters
## -Logistic Regression: Regularization Parameter (C)
best_C = train_C(X_train,y_train)

## -KNN (K-Nearest Neighbors: Number neighbors (K)
best_k = train_k(X_train,y_train)

## -Naibe Bayes : Band width of the Kernel Density (band_width)
best_bd = train_bd(X_train,y_train)

scores_df, y_preds, cms = summarize_scores(X_train,y_train,X_test,y_test)
print("\nSummary of the Scores:\n")
print(scores_df)

#McNemar Test:
summary_nemar = summarize_nemar(cms, y_preds)
print("\nSummary of the McNemar's Test:\n")
print(summary_nemar)





