from methodes.Model import Model
from sklearn.svm import SVC


class SVMClassifier(Model):
    def __init__(self):
        super().__init__(SVC())
        self.name = "SVM"
