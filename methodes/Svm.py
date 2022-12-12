from methodes.Model import Model
from sklearn.svm import SVC


class SVMClassifier(Model):
    def __init__(self):
        super().__init__(SVC(), "SVM")
        self.params_grid = {'kernel': ["poly","rbf","sigmoid"],
                            'C': [0.1, 1, 10, 100, 1000],
                            'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
