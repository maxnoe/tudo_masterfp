import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
plt.style.use('ggplot')
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import metrics
from sklearn import calibration
import sklearn.feature_selection
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from preparation import read_data, drop_useless


def welch_test(a, b, significance=0.05):
    _, p = stats.ttest_ind(a, b, equal_var=False)
    if p < significance:
        print('Null hypothesis rejected with p: {}.'.format(p))
        print('ROC AUC differs significantly')
    else:
        print('Null hypothesis cannot be rejected with p: {}'.format(p))


def classifier_crossval_performance(
        X, y, classifier=GaussianNB(), n_folds=10, bins=50
        ):

    # create axis and figure
    fig, (ax, ax2, ax3) = plt.subplots(3, 1)
    fig.set_size_inches(12, 18)
    # create inset axis for zooming
    axins = zoomed_inset_axes(ax, 3.5, loc=1)

    labels_predictions = []

    # save all aucs and confusion matrices for each cv fold
    roc_aucs = []
    confusion_matrices = []
    precisions = []
    recalls = []
    f_scores = []

    # iterate over test and training sets
    cv = cross_validation.StratifiedKFold(y, n_folds=n_folds)

    for train, test in tqdm(cv):
        # select data
        xtrain, xtest = X[train], X[test]
        ytrain, ytest = y[train], y[test]

        # fit and predict
        classifier.fit(xtrain, ytrain)
        y_probas = classifier.predict_proba(xtest)[:, 1]
        y_prediction = classifier.predict(xtest)
        labels_predictions.append((ytest, y_prediction, y_probas))

    # calculate metrics
    # save all aucs and confusion matrices for each cv fold
    roc_aucs = np.zeros(n_folds)
    confusion_matrices = np.zeros([n_folds, 2, 2])
    precisions = np.zeros(n_folds)
    recalls = np.zeros(n_folds)
    f_scores = np.zeros(n_folds)

    for i, (test, prediction, proba) in enumerate(labels_predictions):
        matrix = metrics.confusion_matrix(test, prediction)
        p, r, f, _ = metrics.precision_recall_fscore_support(
            test, prediction,
        )
        auc = metrics.roc_auc_score(test, proba)

        confusion_matrices[i] = matrix
        roc_aucs[i] = auc
        precisions[i] = p[1]
        recalls[i] = r[1]
        f_scores[i] = f[1]

    # plot roc aucs
    for test, prediction, proba in labels_predictions:
        fpr, tpr, thresholds = metrics.roc_curve(
            test, proba
        )
        ax.plot(fpr, tpr, linestyle='-', color='k', alpha=0.3)
        axins.plot(fpr, tpr, linestyle='-', color='k', alpha=0.3)

    # plot stuff with confidence cuts
    matrices = np.zeros((n_folds, bins, 2, 2))
    for fold, (test, prediction, probas) in enumerate(labels_predictions):
        for i, cut in enumerate(np.linspace(0, 1, bins)):
            cutted_prediction = prediction.copy()
            cutted_prediction[probas < cut] = 0
            cutted_prediction[probas >= cut] = 1
            confusion = metrics.confusion_matrix(test, cutted_prediction)
            matrices[fold][i] = confusion

    b = np.linspace(0, 1, bins)

    tps = matrices[:, :, 0, 0]
    fns = matrices[:, :, 0, 1]
    fps = matrices[:, :, 1, 0]
    tns = matrices[:, :, 1, 1]

    q_mean = np.mean(tps / np.sqrt(tns), axis=0)
    q_err = np.std(tps / np.sqrt(tns), axis=0)
    e_mean = np.mean(tps / np.sqrt(tns + tps), axis=0)
    e_err = np.std(tps / np.sqrt(tns + tps), axis=0)

    ax2.plot(b, q_mean, 'b+', label=r'$\frac{tps}{\sqrt{tns}}$')
    ax2.fill_between(
        b, q_mean + q_err*0.5, q_mean - q_err*0.5, facecolor='gray', alpha=0.4
    )
    ax2.plot(
        b, e_mean,
        color='#58BADB', linestyle='', marker='+',
        label=r'$\frac{tps}{\sqrt{tns + tps}}$',
    )
    ax2.fill_between(
        b, e_mean + e_err*0.5, e_mean - e_err*0.5, facecolor='gray', alpha=0.4
    )
    ax2.legend(loc='best', fancybox=True, framealpha=0.5)
    ax2.set_xlabel('prediction threshold')

    accs = (tps + tns)/(tps + fps + fns + tns)
    acc_mean = np.mean(accs, axis=0)
    acc_err = np.std(accs, axis=0)

    ax3.plot(b, acc_mean, 'r+', label=r'Accuracy')
    ax3.fill_between(
        b, acc_mean + acc_err * 0.5, acc_mean - acc_err*0.5,
        facecolor='gray', alpha=0.4,
    )
    ax3.legend(loc='best', fancybox=True, framealpha=0.5)
    ax3.set_xlabel('prediction threshold')

    ax.set_xlabel('False Positiv Rate')
    ax.set_ylabel('True Positiv Rate')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    axins.set_xlim(0, 0.15)
    axins.set_ylim(0.8, 1.0)
    axins.set_xticks([0.0, 0.05, 0.1, 0.15])
    axins.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    mark_inset(ax, axins, loc1=2, loc2=3, fc='none', ec='0.8')
    name = str(type(classifier).__name__)
    ax.set_title('RoC curves for the {} classifier'.format(name), y=1.03)

    print_performance(
        roc_aucs, confusion_matrices, precisions, recalls, f_scores
    )
    plt.show()
    return roc_aucs


def print_performance(
        roc_aucs,
        confusion_matrices,
        precisions,
        recalls,
        f_scores,
        ):

    tp = confusion_matrices[:, 0, 0]
    fn = confusion_matrices[:, 0, 1]
    fp = confusion_matrices[:, 1, 0]
    tn = confusion_matrices[:, 1, 1]

    print('''Confusion Matrix:
        {:>8.2f} +- {:>8.2f}  {:>8.2f} +- {:>8.2f}
        {:>8.2f} +- {:>8.2f}  {:>8.2f} +- {:>8.2f}
    '''.format(
        tp.mean(), tp.std(),
        fn.mean(), fn.std(),
        fp.mean(), fp.std(),
        tn.mean(), tn.std(),
    ))

    fpr = fp / (fp + tn)
    relative_error = (fpr.std() / fpr.mean()) * 100
    print('Mean False Positive Rate: ')
    print('{:.5f} +- {:.5f} (+- {:.1f} %)'.format(
        fpr.mean(), fpr.std(), relative_error
    ))

    print('Mean area under ROC curve: ')
    relative_error = (roc_aucs.std() / roc_aucs.mean()) * 100
    print('{:.5f} +- {:.5f} (+- {:.1f} %)'.format(
        roc_aucs.mean(), roc_aucs.std(), relative_error
    ))

    print('Mean recall:')
    relative_error = (recalls.std() / recalls.mean()) * 100
    print('{:.5f} +- {:.5f} (+- {:.1f} %)'.format(
        recalls.mean(), recalls.std(), relative_error
    ))

    print('Mean fscore:')
    relative_error = (f_scores.std() / f_scores.mean()) * 100
    print('{:.5f} +- {:.5f} (+- {:.1f} %)'.format(
        f_scores.mean(), f_scores.std(), relative_error
    ))


def plot_recall_precision_curve(
        y, y_prediction, bins=50, outputfile=None
        ):
    print(y)
    precision, recall, thresholds = metrics.precision_recall_curve(
        y, y_prediction
    )
    thresholds = np.append(thresholds, 1)

    fraction_of_positives, mean_predicted_value = calibration.calibration_curve(
        y, y_prediction, n_bins=bins
    )
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 5))
    delta = 1 / bins
    bins = np.linspace(0 - delta / 2, 1 + delta / 2, bins)
    ax1.hist(y_prediction[y == 1], bins=bins, histtype='step', label='signal')
    ax1.hist(y_prediction[y == 0], bins=bins, histtype='step', label='background')
    ax1.set_xlim(-delta, 1 + delta)
    ax1.legend(loc='upper center')
    ax1.set_xlabel('Probabilities')

    ax2.plot(thresholds, recall, label='Recall', linestyle='-')
    ax2.plot(thresholds, precision, label='Precision', linestyle='-')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(-0.05, 1.05)
    ax2.legend(loc='lower center')
    ax2.set_xlabel('Confidence Threshold')

    ax3.plot(mean_predicted_value, fraction_of_positives)
    ax3.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax3.set_xlabel('Mean Predicted Value')
    ax3.set_ylabel('Fraction of positives')

    fig.tight_layout(pad=0)
    if outputfile:
        fig.savefig(outputfile)
    else:
        plt.show()


if __name__ == '__main__':

    df = read_data('signal.csv', 'background.csv')

    df = drop_useless(df)

    print(80*'=')
    print('{:^80}'.format('GaussianNB'))
    print(80*'=')
    gnb = GaussianNB()
    df_nb_label = df.dropna(axis=1)['label']
    df_nb = df.dropna(axis=1).drop('label', axis=1)
    print('{} remaining features after dropping columns NaNs.'.format(
        len(df_nb.columns)
    ))

    nb_aucs = classifier_crossval_performance(
        df_nb.values, df_nb_label.values, classifier=gnb
    )

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        df_nb.values, df_nb_label.values, test_size=0.33
    )

    gnb.fit(X_train, y_train)
    y_prediction = gnb.predict_proba(X_test)[:, 1]

    plot_recall_precision_curve(y_test, y_prediction)

    print(80*'=')
    print('{:^80}'.format('RandomForestClassifier'))
    print(80*'=')
    rf = ensemble.RandomForestClassifier(
        n_jobs=-1, n_estimators=48, criterion='entropy'
    )
    X = df.dropna(axis=1).drop('label', axis=1).values
    y = df.dropna(axis=1)['label'].values
    rf_aucs = classifier_crossval_performance(X, y, classifier=rf, bins=120)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.33
    )

    rf.fit(X_train, y_train)
    y_prediction = rf.predict_proba(X_test)[:, 1]
    plot_recall_precision_curve(y_test, y_prediction, bins=rf.n_estimators + 1)

    print(80*'=')
    print('{:^80}'.format('Calibrated RandomForestClassifier'))
    print(80*'=')
    rf_sigmoid = calibration.CalibratedClassifierCV(rf, cv=10, method='sigmoid')
    rf_sigmoid.fit(X_train, y_train)
    y_prediction = rf_sigmoid.predict_proba(X_test)[:, 1]

    calibrated_rf_aucs = classifier_crossval_performance(
        X, y, classifier=rf_sigmoid, bins=rf.n_estimators + 1
    )
    plot_recall_precision_curve(y_test, y_prediction, bins=100)

    print(80*'=')
    print('{:^80}'.format('ExtraTreesClassifier'))
    print(80*'=')
    extra_rf = ensemble.ExtraTreesClassifier(
        n_jobs=-1, n_estimators=48, criterion='entropy'
    )
    X = df.dropna(axis=1).drop('label', axis=1).values
    y = df.dropna(axis=1)['label'].values
    erf_aucs = classifier_crossval_performance(
        X, y, classifier=extra_rf, bins=120
    )

    X_train, X_test, y_train, y_test, = cross_validation.train_test_split(
        X, y, test_size=0.33
    )
    extra_rf.fit(X_train, y_train)

    y_prediction = extra_rf.predict_proba(X_test)[:, 1]
    plot_recall_precision_curve(
        y_test, y_prediction, bins=extra_rf.n_estimators + 1
    )

    print(80*'=')
    print('{:^80}'.format('Calibrated ExtraTreesClassifier'))
    print(80*'=')
    extra_rf_sigmoid = calibration.CalibratedClassifierCV(
        extra_rf, cv=10, method='sigmoid'
    )
    calibrated_erf_aucs = classifier_crossval_performance(
        X, y, classifier=extra_rf_sigmoid, bins=120
    )

    extra_rf_sigmoid.fit(X_train, y_train)
    y_prediction = extra_rf_sigmoid.predict_proba(X_test)[:, 1]
    plot_recall_precision_curve(
        y_test, y_prediction, bins=extra_rf.n_estimators + 1
    )

    print(80*'=')
    print('{:^80}'.format('GradientBoostingClassifier'))
    print(80*'=')
    gbc = ensemble.GradientBoostingClassifier(n_estimators=10, max_depth=3)
    X = df.dropna(axis=1).drop('label', axis=1).values
    y = df.dropna(axis=1)['label'].values
    grb_aucs = classifier_crossval_performance(X, y, classifier=gbc)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.33
    )
    gbc.fit(X_train, y_train)

    y_prediction = gbc.predict_proba(X_test)[:, 1]
    plot_recall_precision_curve(y_test, y_prediction)

    print('Compare Naive Bayes to Random Forest')
    welch_test(nb_aucs, rf_aucs)

    print('Compare Random Forest to Extremly Random Forest')
    welch_test(rf_aucs, erf_aucs)

    print('Compare Extremly Random Forest to Gradient Boosting Classifier')
    welch_test(erf_aucs, grb_aucs)

    X = df.dropna(axis=1).drop('label', axis=1)
    y = df.dropna(axis=1)['label']

    rfe = sklearn.feature_selection.RFE(LogisticRegression(), 10, step=25)
    X_sel = rfe.fit_transform(X, y)

    print('Selected features: ')
    print(rfe.ranking_)

    fs_extra_rf = ensemble.ExtraTreesClassifier(
        n_jobs=-1, n_estimators=48, criterion='entropy'
    )
    fs_erf_aucs = classifier_crossval_performance(
        X_sel, y.values, classifier=fs_extra_rf, bins=120
    )

    print('Compare GNB to Features Slected GNB')
    welch_test(fs_erf_aucs, erf_aucs)
