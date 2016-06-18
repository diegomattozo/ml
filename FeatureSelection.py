from sklearn.cross_validation import train_test_split
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score


class BackWardSelector(object):

    def __init__(self, estimator, n_features, score_method=accuracy_score, test_size=0.2, random_state=1):
        self.estimator = estimator
        self.n_features = n_features # minimum number of features.
        self.score_method = score_method
        self.test_size = test_size
        self.random_state = random_state
        self.scores = []
        self.models = []

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        dim = X_train.shape[1]
        indexes = tuple(range(dim))  # tuple: (1, 2, 3, ..., k) -> k initial features in our model.
        score = self._calc_score(X_train, y_train, X_test, y_test, indexes)
        self.scores.append(score)
        self.models.append(indexes)

        while dim > self.n_features:
            scores = []
            models = []
            for model in combinations(indexes, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, model)
                scores.append(score)
                models.append(model)

            best_model_index = np.argmax(scores)
            indexes = models[best_model_index]
            self.scores.append(scores[best_model_index])
            self.models.append(models[best_model_index])
            dim -= 1
        return self

    def transform(self, X, index):
        return X[:self.models[index]]

    def _calc_score(self, X_train, y_train, X_test, y_test, model):
        self.estimator.fit(X_train[:, model], y_train)
        y_pred = self.estimator.predict(X_test[:, model])
        score = self.score_method(y_test, y_pred)
        return score


# Usage:
# knn = KNeighborsClassifier(n_neighbors=2)
# sbs = BackWardSelector(knn, n_features=1)
# sbs.fit(X_train_std, y_train)
# print(sbs.models) #return best models.


