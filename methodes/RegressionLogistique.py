from methodes.Model import Model
from sklearn.linear_model import LogisticRegression


class RegressionLogistique(Model):
    def __init__(self):
        super().__init__(LogisticRegression(),"Logistic Regression")
        self.params_grid={'solver':["lbfgs","newton-cg"],
                            'C': [0.5,1,5,10,15,20],
                            'max_iter':[100]}

