import pandas as pd
import numpy as np
import os

from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
# from IPython import embed
from sklearn.naive_bayes import GaussianNB

from preparation import read_data, drop_useless
from tqdm import tqdm


def crossval(X, y, classifier, n_folds=10):

    cval = StratifiedKFold(y, n_folds=n_folds, shuffle=True)

    thresholds = np.linspace(0, 1.0, 101)

    performances = {}

    for n_cv, (train, test) in tqdm(enumerate(cval)):
        performance = pd.DataFrame()

        classifier.fit(X[train], y[train])

        proba = classifier.predict_proba(X[test])[:, 1]

        tp = np.zeros_like(thresholds)
        fp = np.zeros_like(thresholds)
        fn = np.zeros_like(thresholds)
        tn = np.zeros_like(thresholds)
        for i, threshold in enumerate(thresholds):
            tp[i] = np.sum((y[test] == 1) & (proba >= threshold))
            fp[i] = np.sum((y[test] == 0) & (proba >= threshold))
            fn[i] = np.sum((y[test] == 1) & (proba < threshold))
            tn[i] = np.sum((y[test] == 0) & (proba < threshold))

        performance['threshold'] = thresholds
        performance['tpr'] = tp / (tp + fn)
        performance['fpr'] = fp / (fp + tn)
        performance['precision'] = tp / (tp + fp)
        performance['accuracy'] = (tp + tn) / (tp + fp + tn + fn)

        performances[n_cv] = performance

    performances = pd.Panel(performances)

    return performances


data = drop_useless(read_data('./signal.csv', './background.csv'))

nb = GaussianNB()
classifiers = {
    'RandomForest': RandomForestClassifier(
        n_estimators=10, criterion='entropy', n_jobs=-1
    ),
    # 'ExtraTrees': ExtraTreesClassifier(
    #     n_estimators=100, criterion='entropy', n_jobs=-1
    # ),
    'AdaBoost': GradientBoostingClassifier(
        n_estimators=10, loss='exponential',
    ),
    'NaiveBayes': GaussianNB()
}

n_crossval = 3
X = data.drop('label', axis=1).values
y = data['label'].values
for name, classifier in classifiers.items():
    print(name)
    performances = crossval(X, y, classifier, n_crossval)
    performances.to_hdf(os.path.join('build/', name + '.hdf5'), 'performance')

model = SelectFromModel(classifiers['RandomForest'], prefit=True)
X_reduced = model.transform(X)
print('Feature selection selected {} columns'.format(X_reduced.shape[1]))
for name, classifier in classifiers.items():
    print(name)
    performances_reduced = crossval(X_reduced, y, classifier, n_crossval)
    performances_reduced.to_hdf(os.path.join('build/', name + '.hdf5'), 'performance_feature_selection')
