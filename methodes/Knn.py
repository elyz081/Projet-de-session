from sklearn.neighbors import KNeighborsClassifier
from methodes.Model import Model


class Knn(Model):
    def __init__(self):
        super().__init__(KNeighborsClassifier(), "Knn")
        self.params_grid = {'n_neighbors': (1,10, 1),
                            'leaf_size': (20,40,1),
                            'p': (1,2),
                            'weights': ('uniform', 'distance'),
                            'metric': ('minkowski', 'chebyshev')}


