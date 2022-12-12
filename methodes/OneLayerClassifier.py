from sklearn.linear_model import Perceptron
from methodes.Model import Model


class OneLayerClassifier(Model):
    def __init__(self):
        super().__init__(Perceptron())
        self.name = "Perceptron"
