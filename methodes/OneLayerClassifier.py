from sklearn.linear_model import Perceptron
from methodes.Model import Model


class OneLayerClassifier(Model):
    def __init__(self):
        super().__init__(Perceptron(), "One Layer")
        self.params_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1],
                            'n_iter': [5, 10, 15, 20, 50]}
