from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
class Model:
    def __init__(self, model,name):
        self.model = model
        self.name = name
        self.best_model=0
        self.score_before=0
        self.score_after=0


    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)


    def tuning(self,param_grid,X_train, y_train):

        print("# Searching for hyper-parameters for %s" %self.name)
        clf = GridSearchCV(self.model,param_grid, cv=5,scoring='accuracy')
        clf.fit(X_train, y_train)

        print("Score Grid:")
        score = clf.cv_results_['mean_test_score']
        for test_score, params in zip(score, clf.cv_results_['params']):
                print("For %r : /n The accuracy is %0.3f " % (params,test_score))
        
        print("-------------Best hyper parameters :------------")
        print(clf.best_params_)
        self.best_model = clf
        self.score_after=clf.best_score_

    
    def crossed_validation(self, X_train, y_train):   
  
            print("------------Crossed Validation :--------------- ")
            sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=21) 
            i=1
            meanScoreCV = 0
            for train_index, test_index in sss.split(X_train, y_train): 
                i += 1
                X_train_fold, X_test_fold = X_train.values[train_index], X_train.values[test_index]
                y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
                self.train(X_train_fold,y_train_fold)
                y_pred_fold = self.predict(X_test_fold)
                score = accuracy_score(y_test_fold,y_pred_fold)
                meanScoreCV += score
                print("Accuracy of the %dth iteration:" %i,score)

            meanScoreCV /= i-1
            print("The cross-validation of the basic model shows an average score of %.2f" %(meanScoreCV))
            self.score_before=meanScoreCV