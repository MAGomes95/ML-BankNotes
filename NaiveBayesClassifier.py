import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity


class NaiveBayesClassifier:

    def __init__(self, band_value=1.0):

        self.band = band_value
        self.y_log_probabilities = list()
        self.probabilities_vector = dict()

    def fit(self, X, y):
        self.y_log_probabilities = np.log([((y==i).sum())/len(y) for i in np.sort(np.unique(y))])


        for idf in range(X.shape[1]):
            for idt in np.sort(np.unique(y)):
                # Ver depois
                kde = KernelDensity(bandwidth=self.band)
                self.probabilities_vector[idf, idt] = kde

        for keys in self.probabilities_vector.keys():
            self.probabilities_vector[keys] = self.probabilities_vector[keys].fit(
                X[np.where(y == keys[1])][:, keys[0]].reshape(-1, 1))

        return self

    def predict_probs(self, X):

        prob_0, prob_1 = 0, 0

        for feature in range(X.shape[1]):
            prob_0 += self.probabilities_vector[feature, 0].score_samples(X[:, feature].reshape(-1, 1))
            prob_1 += self.probabilities_vector[feature, 1].score_samples(X[:, feature].reshape(-1, 1))
        prob_0 += self.y_log_probabilities[0]
        prob_1 += self.y_log_probabilities[1]

        result = np.column_stack((tuple(np.exp(prob_0)), tuple(np.exp(prob_1))))
        return (result)

    def predict(self, X):

        result = list()
        for list_ in self.predict_probs(X):
            result.append(np.argmax(list_))
        return (np.array(result))

    def score(self, y_true, y_pred):
        return (accuracy_score(y_true, y_pred))

