from sklearn.ensemble import RandomForestClassifier
from methodes.Model import Model


class ForetDecision(Model):

    def __init__(self):
        super().__init__(RandomForestClassifier())
        self.name = "Foret Decision"