import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class GestionDonnees:
    def __init__(self):
        self.train = "data/train.csv"
        self.test = "data/test.csv"
        self.species = []

    def charger_donnees(self, normal=False):
        """

        :param normal: scalling data
        :return: X_train, Y_train
        """
        train = pd.read_csv(self.train) 
        le = LabelEncoder().fit(train.species) 
        self.species = list(le.classes_)
        y_train = le.transform(train.species)                                         
        X_train = train.drop(['species', 'id'], axis=1) 
        if normal:
            X_train = StandardScaler().fit_transform(X_train) 
            X_train=pd.DataFrame(X_train)
        return X_train,y_train

    def charger_test(self,normal=False):
        """

        :param normal: scalling data
        :return: X_test, id_test
        """
        test = pd.read_csv(self.test) 
        id_test = test.id    
        X_test = test.drop(['id'], axis=1)
        if normal:
            X_test = StandardScaler().fit_transform(X_test)
        return X_test,id_test

    