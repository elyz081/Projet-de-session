from sklearn.tree import DecisionTreeClassifier
from methodes.Model import Model


class ArbreDecision(Model):
    def __init__(self):
        super().__init__(DecisionTreeClassifier(), "Decision Tree")
        self.params_grid = {'criterion': ["entropy", "gini"],
                            'max_depth': [2, 3, 4, 5, 6, 10]}
