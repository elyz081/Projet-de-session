from sklearn.neighbors import KNeighborsClassifier
from methodes.Model import Model


class Knn(Model):
    def __init__(self):
        super().__init__(KNeighborsClassifier())
        self.name = "Knn"
