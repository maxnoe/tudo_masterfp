import pandas as pd
import numpy as np
import os

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB

from preparation import read_data, drop_useless
from tqdm import tqdm


def crossval(classifier, df, n_folds=10):

    X = df.drop('label', axis=1).values
    y = df['label'].values

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

classifiers = {
    'RandomForest': RandomForestClassifier(
        n_estimators=100, criterion='entropy', n_jobs=-1
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=100, criterion='entropy', n_jobs=-1
    ),
    'AdaBoost': GradientBoostingClassifier(
        n_estimators=100, loss='exponential',
    ),
    'NaiveBayes': GaussianNB()
}

for name, classifier in classifiers.items():
    print(name)
    performances = crossval(classifier, data, 10)
    performances.to_hdf(os.path.join('build/', name + '.hdf5'), 'performance')
