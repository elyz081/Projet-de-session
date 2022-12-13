from sklearn.ensemble import RandomForestClassifier
from methodes.Model import Model


class ForetDecision(Model):

    def __init__(self):
        super().__init__(RandomForestClassifier(), "Random Forest")
        self.params_grid = {'bootstrap': [True, False],
                             'max_depth': [10, 20, 30, 40, None],
                             'max_features': ['auto', 'sqrt'],
                             'n_estimators': [20,60,100]}

