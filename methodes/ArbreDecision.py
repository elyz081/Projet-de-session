from sklearn.tree import DecisionTreeClassifier
from methodes.Model import Model


class ArbreDecision(Model):
    def __init__(self):
        super().__init__(DecisionTreeClassifier())
        self.name = "Arbre Decision"