from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
class Model:
    def __init__(self, model,name):
        self.model = model
        self.name = name


    def train(self,X_train,Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, pred, actual):
        return metrics.accuracy_score(actual, pred)

    def tuning(self,param_grid,X_train, y_train):

        print("# Réglage des hyper-parametres pour %s" %self.name)

        clf = GridSearchCV(self.model,param_grid, cv=5,scoring='accuracy')
        clf.fit(X_train, y_train)
        print("Grille des scores :")
        score = clf.cv_results_['mean_test_score']
        for test_score, params in zip(score, clf.cv_results_['params']):
                print("For %r : /n The accuracy of Validation Set %0.3f " % (params,test_score))
        
        print("-------------Meilleurs hyper parametres trouvés :------------")
        print(clf.best_params_)
        self.model = clf

    
    def crossed_validation(self, X_train, y_train):   
        #Création de plusieurs divisons de l'ensemble des donnees     
            print(" --- Validation croisee --- ")
            sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=21) #n_spilt divisions, 0.2 = test 0.8 = train
            i=1
            meanScoreCV = 0
            for train_index, test_index in sss.split(X_train, y_train): 
                print("Iteration ",i)
                i += 1
                X_train_fold, X_test_fold = X_train.values[train_index], X_train.values[test_index]
                y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

                #Prediction pour chaque partie
                y_pred_fold = self.predict(X_test_fold)

                #Récupération du score
                score = accuracy_score(y_test_fold,y_pred_fold)
                meanScoreCV += score
                print("Accuracy of the %d eme iteration:" %i,score)

            #Score moyen pour toutes les divisions
            meanScoreCV /= i-1
            print("La validation croisée sans recherche d'hyperparametre affiche un score moyen de test  de %.2f" %(meanScoreCV))
            return meanScoreCV