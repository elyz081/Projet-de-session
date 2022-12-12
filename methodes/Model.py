class Model:
    def __init__(self, model):
        self.model = model


    def train(self,X_train,Y_train):
        self.model.fit(self.X_train, self.Y_train)

    def predict(self, X):
        return self.model.predict(X)

    def erreur(self, t, prediction): 
        return 1
