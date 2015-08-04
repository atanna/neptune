from collections import defaultdict
from sklearn import ensemble, cross_validation
import numpy as np
import scipy as sp
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.lda import LDA
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, \
    accuracy_score, roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, BaggingClassifier, BaggingRegressor, RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier as RForestClass
from sklearn.ensemble import RandomForestRegressor as RForestRegress
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, \
    f_regression, VarianceThreshold, SelectFpr, SelectFwe, SelectFdr
from sklearn.feature_selection import chi2
import operator
import copy
from sklearn.preprocessing import Imputer, Normalizer
from sklearn.svm import LinearSVC
from lib.scores import bac_metric


class OurAutoML:
    """
    Code from run.py in sample
    """
    def __init__(self, info):
        self.target_num = info['target_num']
        self.task = info['task']
        self.metric = info['metric']
        self.sparse = info['is_sparse']
        self.task = info['task']
        self.selector = None
        self.y_density = None
        self.best_clf = None

    def fit(self, X_train, Y_train, n_estimators=100):
        if self.task == 'binary.classification' or self.task == 'multiclass.classification':
            self._binary_classifier(X_train, Y_train, n_estimators=n_estimators)

        elif self.task == 'multilabel.classification':
            if self.sparse:
                self.Ms = [BaggingClassifier(base_estimator=BernoulliNB(), n_estimators=n_estimators/10, n_jobs=-1).fit(X_train, Y_train[:, i]) for i in range(self.target_num)]
            else:
                self.Ms = [RForestClass(n_estimators, random_state=1, n_jobs=-1).fit(X_train, Y_train[:, i]) for i in range(self.target_num)]
        elif self.task == 'regression':
            if self.sparse:
                self.M = BaggingRegressor(base_estimator=BernoulliNB(), n_estimators=n_estimators/10, n_jobs=-1).fit(X_train, Y_train)
            else:
                self.M = RForestRegress(n_estimators, random_state=n_estimators, n_jobs=-1).fit(X_train, Y_train)
        else:
            assert "task not recognised"
        return self

    def _es_density(self, Y):
        self.y_density = defaultdict(lambda:0)
        w = 1./len(Y)
        for label in set(Y):
            self.y_density[label] = w *sum(Y==label)

    def _get_clf(self, n_estimators, selectors):
        if self.sparse:
                clf = BaggingClassifier(base_estimator=BernoulliNB(),
                                        n_estimators=n_estimators/10, n_jobs=-1)
        else:
            clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        seq_to_pipeline = []
        seq_to_pipeline.append(("-const",VarianceThreshold()))
        for selector in selectors:
            seq_to_pipeline.append(selector)
        seq_to_pipeline.append(("RF",clf))
        return Pipeline(seq_to_pipeline)

    def preprocess_bin_cl(self, X, Y, list_clf=None, n_estimators=50, test_size=0.4):
        X_train, X_test, y_train, y_test = cross_validation \
            .train_test_split(X, Y, test_size=test_size)

        if list_clf is None:
            list_clf = [
                self._get_clf(n_estimators, []),
                self._get_clf(n_estimators, [("SelectFpr",
                                              SelectFpr())]),
                self._get_clf(n_estimators, [("SelectFwe",
                                              SelectFwe())]),
                self._get_clf(n_estimators, [("SelectFdr",
                                              SelectFdr())]),
                self._get_clf(n_estimators, [("SelectPercentile",
                                              SelectPercentile())])
                # self._get_clf(n_estimators, [("PCA", PCA())]),
                # self._get_clf(n_estimators, [("LDA", LDA())]),
                # self._get_clf(n_estimators, [('linearSVC', LinearSVC())])
            ]

        best_auc_clf, best_auc, best_auc__bac = None, 0, 0
        best_bac_clf, best_bac, best_bac__auc = None, 0, 0
        parameters = {'RF__criterion': ('gini', 'entropy'),
                      'RF__max_features': ('auto', 'sqrt'),
                      'RF__max_depth': (50, 100, None)
                      }

        for cl in list_clf:
            # try:
            param = dict(parameters)
            for step in cl.steps:
                if step[0] in {"SelectFpr", "SelectFwe", "SelectFdr"}:
                    param.update({"{}__alpha"
                                 .format(step[0]): (0.01, 0.05, 0.1)})
                if step[0] == "SelectPercentile":
                    param.update(SelectPercentile__percentile=(10, 15, 20, 40, 80))

            gs = GridSearchCV(cl, param,
                              scoring='roc_auc',
                              n_jobs=-1).fit(X_train, y_train)
            cl = gs.best_estimator_
            y_pred = cl.predict_proba(X_test)[:, 1]
            auc_cv = cross_val_score(cl, X_test, y_test, scoring="roc_auc", cv=3, n_jobs=-1)
            auc = roc_auc_score(y_test, y_pred)
            bac = bac_metric(y_test, y_pred)
            print("n_params: {}  ({})".format(cl.transform(X[0:1])
                                              .shape[1], X.shape[1]))
            print("steps: {steps}\n"
                  "auc: {auc} {auc_cv_mean} ({auc_cv})\nbac: {bac}\n"
                  .format(steps=cl.steps,
                          auc=auc,
                          auc_cv_mean=auc_cv.mean(),
                          auc_cv=auc_cv,
                          bac=bac))
            auc = auc_cv.mean()
            if auc > best_auc:
                best_auc_clf, best_auc, best_auc__bac = cl, auc, bac
            if bac > best_bac:
                best_bac_clf, best_bac, best_bac__auc = cl, bac, auc

        self.best_clf = best_auc_clf
        print("best_auc_clf:")
        print("steps:{} auc: {}  bac: {}"
              .format([step[0] for step in best_auc_clf.steps],
                      best_auc,
                      best_auc__bac))

        print("best_bac_clf:")
        print("steps:{} auc: {}  bac: {}"
              .format([step[0] for step in best_bac_clf.steps],
                      best_bac__auc,
                      best_bac))

        d_bac = best_bac - best_auc__bac
        d_auc = best_auc - best_bac__auc

        if d_bac - d_auc > 0:
            self.best_clf = best_bac_clf
        else:
            self.best_clf = best_auc_clf

    def _binary_classifier(self, X, Y, n_estimators=100, min_features=150):
        """
        Main fit function
        """
        self._es_density(Y)
        if self.sparse:
                clf = BaggingClassifier(base_estimator=BernoulliNB(),
                                        n_estimators=n_estimators/10, n_jobs=-1)
        else:
            # clf = RForestClass(n_estimators, n_jobs=-1)
            clf = RandomForestClassifier(n_estimators=n_estimators,
                                         criterion='entropy',

                                         n_jobs=-1)

        if self.best_clf is not None:
            self.M = self.best_clf.fit(X,Y)
            return

        n_features = X.shape[1]
        seq_to_pipeline = []
        seq_to_pipeline.append(('discard_const_features', VarianceThreshold()))
        if n_features > min_features:
            seq_to_pipeline.append(("SelectFpr", SelectFpr()))
            # seq_to_pipeline.append(("PCA", PCA()))
            # seq_to_pipeline.append(("LDA", LDA()))
            # seq_to_pipeline.append(('linearSVC', LinearSVC()))
        seq_to_pipeline.append(('clf', clf))
        self.M = Pipeline(seq_to_pipeline)
        self.M.fit(X, Y)
        print("n_params: {}  ({})".format(self.M.transform(X[0:1])
                                          .shape[1], X.shape[1]))

    def fit_and_count_av_score(self, X, Y, cv=3, n_estimators=100, test_size=0.4):
        X_train, X_test, y_train, y_test = cross_validation\
            .train_test_split(X, Y, test_size=test_size)
        self.fit(X_train, y_train, n_estimators)
        auc_scores = cross_val_score(self.M, X_test, y_test, cv=cv, n_jobs=-1,
                                 scoring='roc_auc')
        y_pred = self.predict(X_test)
        print("roc_auc_score", auc_scores.mean())
        print("bac_score: ", bac_metric(y_test, y_pred))
        print(classification_report(y_test, self.bin(y_pred, 0)))

    def predict(self, X):
        if self.task == 'binary.classification':
            Y_pred = self.M.predict_proba(X)[:, 1]
        elif self.task == 'multiclass.classification':
            Y_pred = np.array([self.M.predict_proba(X)[:, i] for i in range(self.target_num)]).T

        elif self.task == 'multilabel.classification':
            Y_pred = np.array([self.Ms[i].predict_proba(X)[:, 1] for i in range(self.target_num)]).T
        elif self.task == 'regression':
            Y_pred = self.M.predict(X)

        if self.sparse:
            if self.task == 'multilabel.classification' \
                    or self.task == 'multiclass.classification':
                eps = 0.001
                for i in range(len(Y_pred)):
                    pos = np.argmax(Y_pred[i])
                    Y_pred[i] += eps
                    Y_pred[i][pos] -= self.target_num * eps
        return Y_pred

    def bin(self, Y, eps=1e-6):
        if self.y_density is not None and len(self.y_density) == 2:
            threshold = self.y_density[0] / (self.y_density[0] + self.y_density[1])
        Y[Y < threshold] = eps
        Y[Y >= threshold] = 1. - eps
        return Y




class Scores():
    def __init__(self, y_true, y_pred):
        pass

    @staticmethod
    def bac(y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        sensitivity = tpr
        specificity = 1-fpr
        return (sensitivity + specificity) / 2




#it's not use now. Just sample
class MyAutoML:
    ''' Rough sketch of a class that "solves" the AutoML problem. We illustrate various type of data that will be encountered in the challenge can be handled.
         Also, we make sure that the model regularly outputs predictions on validation and test data, such that, if the execution of the program is interrupted (timeout)
         there are still results provided by the program. The baseline methods chosen are not optimized and do not provide particularly good results.
         In particular, no special effort was put into dealing with missing values and categorical variables.
         
         The constructor selects a model based on the data information passed as argument. This is a form of model selection "filter".
         We anticipate that the participants may compute a wider range of statistics to perform filter model selection.
         We also anticipate that the participants will conduct cross-validation experiments to further select amoung various models
         and hyper-parameters of the model. They might walk trough "model space" systematically (e.g. with grid search), heuristically (e.g. with greedy strategies),
         or stochastically (random walks). This example does not bother doing that. We simply use a growing ensemble of models to improve predictions over time.
         
         We use ensemble methods that vote on an increasing number of classifiers. For efficiency, we use WARM START that re-uses
         already trained base predictors, when available.
         
        IMPORTANT: This is just a "toy" example:
            - if was checked only on the phase 0 data at the time of release
            - not all cases are considered
            - this could easily break on datasets from further phases
            - this is very inefficient (most ensembles have no "warm start" option, hence we do a lot of unnecessary calculations)
            - there is no preprocessing
         '''
         
    def __init__(self, info, verbose=True, debug_mode=False):
        self.label_num=info['label_num']
        self.target_num=info['target_num']
        self.task = info['task']
        self.metric = info['metric']
        self.postprocessor = None
        #self.postprocessor = MultiLabelEnsemble(LogisticRegression(), balance=True) # To calibrate proba
        self.postprocessor = MultiLabelEnsemble(LogisticRegression(), balance=False) # To calibrate proba
        if debug_mode>=2:
            self.name = "RandomPredictor"
            self.model = RandomPredictor(self.target_num)
            self.predict_method = self.model.predict_proba 
            return
        if info['task']=='regression':
            if info['is_sparse']==True:
                self.name = "BaggingRidgeRegressor"
                self.model = BaggingRegressor(base_estimator=Ridge(), n_estimators=1, verbose=verbose) # unfortunately, no warm start...
            else:
                self.name = "GradientBoostingRegressor"
                self.model = GradientBoostingRegressor(n_estimators=1, verbose=verbose, warm_start = True)
            self.predict_method = self.model.predict # Always predict probabilities
        else:
            if info['has_categorical']: # Out of lazziness, we do not convert categorical variables...
                self.name = "RandomForestClassifier"
                self.model = RandomForestClassifier(n_estimators=1, verbose=verbose) # unfortunately, no warm start...
            elif info['is_sparse']:                
                self.name = "BaggingNBClassifier"
                self.model = BaggingClassifier(base_estimator=BernoulliNB(), n_estimators=1, verbose=verbose) # unfortunately, no warm start...                          
            else:
                self.name = "GradientBoostingClassifier"
                self.model = eval(self.name + "(n_estimators=1, verbose=" + str(verbose) + ", min_samples_split=10, random_state=1, warm_start = True)")
            if info['task']=='multilabel.classification':
                self.model = MultiLabelEnsemble(self.model)
            self.predict_method = self.model.predict_proba  
                          

    def __repr__(self):
        return "MyAutoML : " + self.name

    def __str__(self):
        return "MyAutoML : \n" + str(self.model) 

    def fit(self, X, Y):
        self.model.fit(X,Y)
        # Train a calibration model postprocessor
        if self.task != 'regression' and self.postprocessor!=None:
            Yhat = self.predict_method(X)
            if len(Yhat.shape)==1: # IG modif Feb3 2015
                Yhat = np.reshape(Yhat,(-1,1))           
            self.postprocessor.fit(Yhat, Y)
        return self
        
    def predict(self, X):
        prediction = self.predict_method(X)
        # Calibrate proba
        if self.task != 'regression' and self.postprocessor!=None:          
            prediction = self.postprocessor.predict_proba(prediction)
        # Keep only 2nd column because the second one is 1-first    
        if self.target_num==1 and len(prediction.shape)>1 and prediction.shape[1]>1:
            prediction = prediction[:,1]
        # Make sure the normalization is correct
        if self.task=='multiclass.classification':
            eps = 1e-15
            norma = np.sum(prediction, axis=1)
            for k in range(prediction.shape[0]):
                prediction[k,:] /= sp.maximum(norma[k], eps)  
        return prediction


class MultiLabelEnsemble:
    ''' MultiLabelEnsemble(predictorInstance, balance=False)
        Like OneVsRestClassifier: Wrapping class to train multiple models when 
        several objectives are given as target values. Its predictor may be an ensemble.
        This class can be used to create a one-vs-rest classifier from multiple 0/1 labels
        to treat a multi-label problem or to create a one-vs-rest classifier from
        a categorical target variable.
        Arguments:
            predictorInstance -- A predictor instance is passed as argument (be careful, you must instantiate
        the predictor class before passing the argument, i.e. end with (), 
        e.g. LogisticRegression().
            balance -- True/False. If True, attempts to re-balance classes in training data
            by including a random sample (without replacement) s.t. the largest class has at most 2 times
        the number of elements of the smallest one.
        Example Usage: mymodel =  MultiLabelEnsemble (GradientBoostingClassifier(), True)'''
	
    def __init__(self, predictorInstance, balance=False):
        self.predictors = [predictorInstance]
        self.n_label = 1
        self.n_target = 1
        self.n_estimators =  1 # for predictors that are ensembles of estimators
        self.balance=balance
        
    def __repr__(self):
        return "MultiLabelEnsemble"

    def __str__(self):
        return "MultiLabelEnsemble : \n" + "\tn_label={}\n".format(self.n_label) + "\tn_target={}\n".format(self.n_target) + "\tn_estimators={}\n".format(self.n_estimators) + str(self.predictors[0])
	
    def fit(self, X, Y):
        if len(Y.shape)==1: 
            Y = np.array([Y]).transpose() # Transform vector into column matrix
            # This is NOT what we want: Y = Y.reshape( -1, 1 ), because Y.shape[1] out of range
        self.n_target = Y.shape[1]                 # Num target values = num col of Y
        self.n_label = len(set(Y.ravel()))         # Num labels = num classes (categories of categorical var if n_target=1 or n_target if labels are binary )
        # Create the right number of copies of the predictor instance
        if len(self.predictors)!=self.n_target:
            predictorInstance = self.predictors[0]
            self.predictors = [predictorInstance]
            for i in range(1,self.n_target):
                self.predictors.append(copy.copy(predictorInstance))
        # Fit all predictors
        for i in range(self.n_target):
            # Update the number of desired prodictos
            if hasattr(self.predictors[i], 'n_estimators'):
                self.predictors[i].n_estimators=self.n_estimators
            # Subsample if desired
            if self.balance:
                pos = Y[:,i]>0
                neg = Y[:,i]<=0
                if sum(pos)<sum(neg): 
                    chosen = pos
                    not_chosen = neg
                else: 
                    chosen = neg
                    not_chosen = pos
                num = sum(chosen)
                idx=filter(lambda(x): x[1]==True, enumerate(not_chosen))
                idx=np.array(zip(*idx)[0])
                np.random.shuffle(idx)
                chosen[idx[0:min(num, len(idx))]]=True
                # Train with chosen samples            
                self.predictors[i].fit(X[chosen,:],Y[chosen,i])
            else:
                self.predictors[i].fit(X,Y[:,i])
        return
		
    def predict_proba(self, X):
        if len(X.shape)==1: # IG modif Feb3 2015
            X = np.reshape(X,(-1,1))   
        prediction = self.predictors[0].predict_proba(X)
        if self.n_label==2:                 # Keep only 1 prediction, 1st column = (1 - 2nd column)
            prediction = prediction[:,1]
        for i in range(1,self.n_target): # More than 1 target, we assume that labels are binary
            new_prediction = self.predictors[i].predict_proba(X)[:,1]
            prediction = np.column_stack((prediction, new_prediction))
        return prediction
		
class RandomPredictor:
    ''' Make random predictions.'''
	
    def __init__(self, target_num):
        self.target_num=target_num
        return
        
    def __repr__(self):
        return "RandomPredictor"

    def __str__(self):
        return "RandomPredictor"
	
    def fit(self, X, Y):
        if len(Y.shape)>1:
            assert(self.target_num==Y.shape[1])
        return self
		
    def predict_proba(self, X):
        prediction = np.random.rand(X.shape[0],self.target_num)
        return prediction			
