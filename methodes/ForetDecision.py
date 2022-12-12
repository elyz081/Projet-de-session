from sklearn.ensemble import RandomForestClassifier
from methodes.Model import Model


class ForetDecision(Model):

    def __init__(self):
        super().__init__(RandomForestClassifier(), "Random Forest")
        self.params_grid = {{'bootstrap': [True, False],
                             'max_depth': [10, 20, 30, 40, 50, None],
                             'max_features': ['auto', 'sqrt'],
                             'min_samples_leaf': [1, 2, 4],
                             'min_samples_split': [2, 5, 10],
                             'n_estimators': [200, 400, 600, 800]}}

