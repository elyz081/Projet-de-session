from sklearn.linear_model import Perceptron
from methodes.Model import Model
import numpy as np


class OneLayerClassifier(Model):
    def __init__(self):
        super().__init__(Perceptron(), "One Layer")
        self.params_grid = {'alpha': np.linspace(0.0001,0.01,15),
                            "penalty":["l2","l1","elasticnet"]}
