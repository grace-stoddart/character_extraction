import numpy as np


from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn import metrics


from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



def combine_features(featuresDir, featureNames, fileNames):

    # joing features for CR chains in each story individually
    featuresStoryAll = []

    for fileName in fileNames:
        featuresStory = np.atleast_2d(np.load(featuresDir + featureNames[0] + '/' + fileName + '.npy'))
            
        for i in range(1, len(featureNames)):
            feature = np.atleast_2d(np.load(featuresDir + featureNames[i] + '/' + fileName + '.npy'))
            featuresStory = np.concatenate((featuresStory, feature))

        featuresStoryAll.append(featuresStory)

    # now join chain features from each story

    featuresAll = featuresStoryAll[0]

    for i in range(1, len(featuresStoryAll)):
        featuresAll = np.concatenate((featuresAll, featuresStoryAll[i]), axis = 1)

    return featuresAll


def combine_features_old(featureNames, featuresDir, corpus = 'ProppLearner', n = 46):
    '''
    Combines specified features into a single array, which can be used by an SVM
    Parameters:
        featureNames - list containing feature names (i.e. feature FILE names) to be combined
        featuresDir - directory where feature files will be found
        n - num stories to go through in corpus
    Returns:
        featuresCombined - np array
    '''

    if corpus == 'ProppLearner':

        offset = 0

        # the first feature from story 1 will start the array
        npFileName = 'story' + str(1) + '.npy'
        featuresAll = np.atleast_2d(np.load(featuresDir + featureNames[0] + '/' + npFileName))

        # iterate through remaining features in story 1 and add to array
        for i in range(1, len(featureNames)):
            feature = np.atleast_2d(np.load(featuresDir + featureNames[i] + '/' + npFileName))
            featuresAll = np.concatenate((featuresAll, feature))

        # iterate through all remaining features in all remaining stories and add to array
        for storyNum in range(2 + offset,(n+1 + offset)):

            if storyNum == 34:
                continue

            npFileName = 'story' + str(storyNum) + '.npy'

            featuresThisStory = np.atleast_2d(np.load(featuresDir + featureNames[0] + '/' + npFileName))

            for i in range(1, len(featureNames)):
                feature = np.atleast_2d(np.load(featuresDir + featureNames[i] + '/' + npFileName))
                featuresThisStory = np.concatenate((featuresThisStory, feature))

            featuresAll = np.concatenate((featuresAll, featuresThisStory), axis = 1)
            

    else:
        print('need to add this corpus to function')

    return featuresAll.transpose()


def param_selection(X, y, setting = 'normal'):
    '''
    Tries all the combinations of the values passed in the dictionary and evaluates the model for each combination using Cross-Validation.
            Parameters:
                    X - data (features)
                    y- labels
                    param_grid - dictionary containing parameter grid

            Returns:
                    params - dict containing best parameters
                    f1Score - f1 score for these parameters
                    clf.cv_results - full results for further analysis
    '''

    param_grid = [
        {
            'model__C': [0.001, 1.e-02, 1.e-01, 1.e+00, 0.5, 1.e+01, 1.e+02, 1.e+03, 1.e+04], 
            'model__gamma': [1.e-02, 1.e-01, 1.e+00,  1.e+01, 1.e+02], 
            'model__kernel': ['rbf'],
            },
        ]
    #############################
    model = svm.SVC()

    if setting == 'over':
        over = SMOTE(sampling_strategy = 1)
        pipeline = Pipeline([('over', over), ('model', model)])

        clf = GridSearchCV(estimator = pipeline, param_grid=param_grid, scoring='f1')
        clf.fit(X, y)
        argmax = np.argmax(clf.cv_results_['mean_test_score'])

    elif setting == 'over_under':
        over = SMOTE(sampling_strategy = 0.8)
        under = RandomUnderSampler(sampling_strategy = 1.0)
        pipeline = Pipeline([('over', over), ('under', under), ('model', model)])

        clf = GridSearchCV(estimator = pipeline, param_grid=param_grid, scoring='f1')
        clf.fit(X, y)
        argmax = np.argmax(clf.cv_results_['mean_test_score'])
    
    else:
        pipeline = Pipeline([('model', model)])
        clf = GridSearchCV(estimator = pipeline, param_grid=param_grid, scoring='f1')
        clf.fit(X, y)
        argmax = np.argmax(clf.cv_results_['mean_test_score'])
    ##################################

    # over = SMOTE(sampling_strategy = 1.0)

    # Xover, yover = over.fit_resample(X, y)


    # svc = svm.SVC()
    # clf = GridSearchCV(svc, param_grid, scoring='f1')
    # clf.fit(Xover, yover)

    # argmax = np.argmax(clf.cv_results_['mean_test_score'])

    return clf.cv_results_['params'][argmax], clf.cv_results_['mean_test_score'][argmax], clf.cv_results_


def train_and_evaluate_model(model, X, y, setting = 'none', k = 10, n = 20, over_sampling_strategy = 1.0, over_under_sampling_strategy = (0.8, 1.0)):
    '''
    Trains and evaluates classifier using repeated stratified k-fold.
    Returns results from sampling strategies: none, over, over + under

    Parameters
        model - the model to train and evaluate
        k - number of folds
        n - number of repeats (default 10)
        X - features
        y - labels
        over_sampling_strategy - default 1.0
        over_under_sampling_strategy - default (0.75, 1.0)
    
    Returns
        results - mean F1 score and all scores in dict
    
    '''
    if setting == 'none':
        steps = [('model', model)]

    elif setting == 'over':
        over = SMOTE(sampling_strategy = over_sampling_strategy)
        steps = [('over', over), ('model', model)]

    else:
        over = SMOTE(sampling_strategy = over_under_sampling_strategy[0])
        under = RandomUnderSampler(sampling_strategy = over_under_sampling_strategy[1])
        steps = [('over', over), ('under', under), ('model', model)]
        

    pipeline = Pipeline(steps=steps)

    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=n,) #random_state=1)


    scores_acc = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    scores_prec = cross_val_score(pipeline, X, y, scoring='precision', cv=cv, n_jobs=-1)
    scores_rec = cross_val_score(pipeline, X, y, scoring='recall', cv=cv, n_jobs=-1)
    scores_f1 = cross_val_score(pipeline, X, y, scoring='f1', cv=cv, n_jobs=-1)

    results = {
        'accuracy':{
            'avg': round(np.mean(scores_acc),2),
            'all': scores_acc
        },
        'precision':{
            'avg': round(np.mean(scores_prec),2),
            'all': scores_prec
        },
        'recall':{
            'avg': round(np.mean(scores_rec),2),
            'all': scores_rec
        },
        'f1':{
            'avg': round(np.mean(scores_f1),2),
            'all': scores_f1
        }

        }
                
    return results




def train_and_evaluate_model_testing(model, X, y, k, setting = 'normal', n = 10, over_sampling_strategy = 1.0, over_under_sampling_strategy = (0.75, 1.0)):
    '''
    Trains and evaluates classifier using repeated stratified k-fold.
    Returns results from sampling strategies: none, over, over + under

    Parameters
        model - the model to train and evaluate
        k - number of folds
        n - number of repeats (default 10)
        X - features
        y - labels
        over_sampling_strategy - default 1.0
        over_under_sampling_strategy - default (0.75, 1.0)
    
    Returns
        results - mean F1 score and all scores in dict
    
    '''
    if setting == 'normal':
        steps = [('model', model)]

    elif setting =='over':
        over = SMOTE(sampling_strategy = over_sampling_strategy)
        steps = [('over', over), ('model', model)]

    else:
        over2 = SMOTE(sampling_strategy = over_under_sampling_strategy[0])
        under = RandomUnderSampler(sampling_strategy = over_under_sampling_strategy[1])
        steps = [('over', over2), ('under', under), ('model', model)]
        

    pipeline = Pipeline(steps=steps)

    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=n,) #random_state=1)

    scores = cross_val_score(pipeline, X, y, scoring='f1', cv=cv, n_jobs=-1)

    results = {'avg': round(np.mean(scores),2), 'all': scores}
        
    return results



