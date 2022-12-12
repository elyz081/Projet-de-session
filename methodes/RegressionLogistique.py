from methodes.Model import Model
from sklearn.linear_model import LogisticRegression

class RegressionLogistique(Model):
    def __init__(self):
        super().__init__(LogisticRegression())
        self.name="Regression Logistique"