import numpy as np
import pandas as pd
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


def welch_test(a, b, significance=0.05):
    _, p = stats.ttest_ind(a, b, equal_var=False)
    if p < significance:
        print('Null hypothesis rejected with p: {}.'.format(p))
        print('ROC AUC differs significantly')
    else:
        print('Null hypothesis cannot be rejected with p: {}'.format(p))


def classifier_crossval_performance(
        X, y, classifier=GaussianNB(), n_folds=10, weights=None, bins=50
        ):

    # create axis and figure
    fig, (ax, ax2, ax3) = plt.subplots(3, 1)
    fig.set_size_inches(12, 18)
    # create inset axis for zooming
    axins = zoomed_inset_axes(ax, 3.5, loc=1)

    labels_predctions = []

    # save all aucs and confusion matrices for each cv fold
    roc_aucs = []
    confusion_matrices = []
    precisions = []
    recalls = []
    f_scores = []

    # iterate over test and training sets
    cv = cross_validation.StratifiedKFold(y, n_folds=n_folds)
    test_weights = None

    for train, test in tqdm(cv):
        # select data
        xtrain, xtest = X[train], X[test]
        ytrain, ytest = y[train], y[test]

        # fit and predict
        classifier.fit(xtrain, ytrain)
        y_probas = classifier.predict_proba(xtest)[:, 1]
        y_prediction = classifier.predict(xtest)
        labels_predctions.append((ytest, y_prediction, y_probas))

    # calculate metrics
    # save all aucs and confusion matrices for each cv fold
    roc_aucs = np.zeros(n_folds)
    confusion_matrices = np.zeros([n_folds, 2, 2])
    precisions = np.zeros(n_folds)
    recalls = np.zeros(n_folds)
    f_scores = np.zeros(n_folds)

    for i, (test, prediction, proba) in enumerate(labels_predctions):
        matrix = metrics.confusion_matrix(test, prediction)
        p, r, f, _ = metrics.precision_recall_fscore_support(
            test, prediction, sample_weight=test_weights
        )
        auc = metrics.roc_auc_score(test, proba, sample_weight=test_weights)

        confusion_matrices[i] = matrix
        roc_aucs[i] = auc
        precisions[i] = p[1]
        recalls[i] = r[1]
        f_scores[i] = f[1]

    # plot roc aucs
    for test, prediction, proba in labels_predctions:
        fpr, tpr, thresholds = metrics.roc_curve(
            test, proba, sample_weight=test_weights
        )
        ax.plot(fpr, tpr, linestyle='-', color='0.4')
        axins.plot(fpr, tpr, linestyle='-', color='0.4')

    # plot stuff with confidence cuts
    matrices = np.zeros((n_folds, bins, 2, 2))
    for fold, (test, prediction, probas) in enumerate(labels_predctions):
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
        {:.2f} +- {:.2f} \t {:.2f} +- {:.2f}
        {:.2f} +- {:.2f} \t {:.2f} +- {:.2f}
    '''.format(
        tp.mean(), tp.std(),
        fn.mean(), fn.std(),
        fp.mean(), fp.std(),
        tn.mean(), tn.std(),
    ))

    fpr = fp / (fp + tn)
    relative_error = (fpr.std() / fpr.mean()) * 100
    print('Mean False Positive Rate: ')
    print('{:.5f} +- {:.5f} \t (+- {:.1f} % )'.format(
        fpr.mean(), fpr.std(), relative_error
    ))

    print('Mean area under ROC curve: ')
    relative_error = (roc_aucs.std() / roc_aucs.mean()) * 100
    print('{:.5f} +- {:.5f} \t (+- {:.1f} %)'.format(
        roc_aucs.mean(), roc_aucs.std(), relative_error
    ))

    print('Mean recall:')
    relative_error = (recalls.std() / recalls.mean()) * 100
    print('{:.5f} +- {:.5f} \t (+- {:.1f} %)'.format(
        recalls.mean(), recalls.std(), relative_error
    ))

    print('Mean fscore:')
    relative_error = (f_scores.std() / f_scores.mean()) * 100
    print('{:.5f} +- {:.5f} \t (+- {:.1f} %)'.format(
        f_scores.mean(), f_scores.std(), relative_error
    ))


def plot_recall_precission_curve(y, y_prediction, weights=None, bins=50):
    precision, recall, thresholds = metrics.precision_recall_curve(
        y, y_prediction, sample_weight=weights
    )

    fraction_of_positives, mean_predicted_value = calibration.calibration_curve(
        y, y_prediction, n_bins=bins
    )
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 5))
    ax1.hist(y_prediction, bins=bins)
    ax1.set_title('Histogram of predicted probabilities')
    ax1.set_xlabel('Probabilities')

    ax2.plot(recall, precision, label='Recall / TPR', linestyle='-')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision/Recall curve')

    ax3.plot(mean_predicted_value, fraction_of_positives)
    ax3.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax3.set_xlabel('Mean Predicted Value')
    ax3.set_ylabel('Fraction of positives')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    df_signal = pd.read_csv('signal.csv', sep=';')
    df_signal.dropna(axis=[1, 0], how='all', inplace=True)
    print('Number of signal Features: {}. Label {}'.format(
        len(df_signal.columns), df_signal.label.iloc[0])
    )

    df_background = pd.read_csv('background.csv', sep=';')
    df_background.dropna(axis=[1, 0], how='all', inplace=True)
    print('Number of background features: {}. Label: {}'.format(
        len(df_background.columns), df_background.label.iloc[0])
    )

    # lets only take columns that appear in both datasets
    df = pd.concat([df_signal, df_background], axis=0, join='inner')

    # match  all columns containing the name 'corsika'
    c = df.filter(regex='((C|c)orsika)(\w*[\._\-]\w*)?').columns
    df = df.drop(c, axis=1)

    # match any column containing header, MC, utc, mjd, date or ID.
    # Im sure there are better regexes for this.
    c = df.filter(regex='\w*[\.\-_]*(((u|U)(t|T)(c|C))|(MC)|((m|M)(j|J)(d|D))|(Weight)|((h|H)eader)|((d|D)ate)|(ID))+\w*[\.\-_]*\w*').columns
    df = df.drop(c, axis=1)

    # drop columns containing only a single value
    df = df.drop(df[df.var() == 0].index, axis=1)
    # some features are weird. Pearsons r is NaN. Better drop those columns
    # corr = df.apply(lambda c: stats.pearsonr(c, df.label)[0])
    # df = df.drop(corr[corr.isnull()].index, axis=1)
    # print(df.columns)
    print('Combined Features: {}'.format(len(df.columns)))

    signal_weight = df_signal['Weight.HoSa'].sum()
    background_weight = df_background['Weight.HoSa'].sum()
    print('Signal Weight: ')
    print(signal_weight)

    print('Background Weight: ')
    print(background_weight)

    print('Ratio:')
    ratio = background_weight / signal_weight
    print(ratio)

    factor = 10000 / ratio
    df_background['Weight.HoSa'] *= factor
    background_weight = df_background['Weight.HoSa'].sum()
    ratio = background_weight / signal_weight

    # normalize
    sample_weights = np.append(
        df_background['Weight.HoSa'].values, df_signal['Weight.HoSa'].values
    )
    sample_weights /= np.abs(sample_weights.max() - sample_weights.min())

    # I haven't quite figured out how to use these weights for evaluating
    # classfiers within sklearn.
    # Feeding these to performance metric functions does nothing.
    # I'm not going to use them further.

    # Gaussian Naive Bayes Classifier
    #
    # Lets try the Gaussian Naive Bayes algorithm first.
    # It assumes that the values of the features are distributed according
    # to a Gaussian. To quote wikipedia:
    #
    # > Let $\mu_c$ be the mean of the values in $x$ associated with class $c$,
    # > and let $\sigma^2_c$ be the variance of the
    # > values in $x$ associated with class $c$.
    # > Then, the probability distribution of some value given a class,
    # > $p(x=v|c)$, can be computed by plugging v into the equation for a
    # > Normal distribution parameterized by
    # > $\mu_c$ and $\sigma^2_c$. That is,
    # > $$p(x=v|c)=\frac{1}{\sqrt{2\pi\sigma^2_c}}\,e^{ -\frac{(v-\mu_c)^2}{2\sigma^2_c} } $$
    #
    # Feature scaling or mean shifting is not needed for NaiveBayes.
    # It would have no effect. All NaNs and Infs have to removed however.

    # In[ ]:

    gnb = GaussianNB()
    df_nb_label = df.dropna(axis=1)['label']
    df_nb = df.dropna(axis=1).drop('label', axis=1)
    print('{} remaining features after dropping columns NaNs.'.format(
        len(df_nb.columns)
    ))

    nb_aucs = classifier_crossval_performance(
        df_nb.values, df_nb_label.values, classifier=gnb, weights=sample_weights
    )

    X_train, X_test, y_train, y_test, _, weights_test = cross_validation.train_test_split(df_nb, df_nb_label, sample_weights, test_size=0.33)

    gnb.fit(X_train, y_train)
    y_prediction = gnb.predict_proba(X_test)[:, 1]

    plot_recall_precission_curve(y_test, y_prediction)

    rf = ensemble.RandomForestClassifier(
        n_jobs=48, n_estimators=48, criterion='entropy'
    )
    X = df.dropna(axis=1).drop('label', axis=1).values
    y = df.dropna(axis=1)['label'].values
    rf_aucs = classifier_crossval_performance(X, y, classifier=rf, bins=120)

    X_train, X_test, y_train, y_test, _, weights_test = cross_validation.train_test_split(X, y, sample_weights, test_size=0.33)

    rf.fit(X_train, y_train)
    y_prediction = rf.predict_proba(X_test)[:, 1]

    plot_recall_precission_curve(y_test, y_prediction, weights=None, bins=100)

    rf_sigmoid = calibration.CalibratedClassifierCV(rf, cv=10, method='sigmoid')
    rf_sigmoid.fit(X_train, y_train)
    y_prediction = rf_sigmoid.predict_proba(X_test)[:, 1]

    plot_recall_precission_curve(y_test, y_prediction, weights=None, bins=100)

    calibrated_rf_aucs = classifier_crossval_performance(
        X, y, classifier=rf_sigmoid, bins=120
    )

    from sklearn import ensemble
    extra_rf = ensemble.ExtraTreesClassifier(
        n_jobs=24, n_estimators=2*24, criterion='entropy'
    )
    X = df.dropna(axis=1).drop('label', axis=1).values
    y = df.dropna(axis=1)['label'].values
    erf_aucs = classifier_crossval_performance(
        X, y, classifier=extra_rf, bins=120
    )

    X_train, X_test, y_train, y_test, _, weights_test = cross_validation.train_test_split(X, y, sample_weights, test_size=0.33)
    extra_rf.fit(X_train, y_train)

    y_prediction = extra_rf.predict_proba(X_test)[:, 1]

    plot_recall_precission_curve(y_test, y_prediction, weights=None)

    extra_rf_sigmoid = calibration.CalibratedClassifierCV(
        extra_rf, cv=10, method='sigmoid'
    )
    extra_rf_sigmoid.fit(X_train, y_train)
    y_prediction = extra_rf_sigmoid.predict_proba(X_test)[:, 1]

    plot_recall_precission_curve(y_test, y_prediction, weights=None, bins=100)

    calibrated_erf_aucs = classifier_crossval_performance(
        X, y, classifier=extra_rf_sigmoid, bins=120
    )

    gbc = ensemble.GradientBoostingClassifier(n_estimators=10, max_depth=3)
    X = df.dropna(axis=1).drop('label', axis=1).values
    y = df.dropna(axis=1)['label'].values
    grb_aucs = classifier_crossval_performance(X, y, classifier=gbc)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.33
    )
    gbc.fit(X_train, y_train)

    y_prediction = gbc.predict_proba(X_test)[:, 1]
    plot_recall_precission_curve(y_test, y_prediction)

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
        n_jobs=24, n_estimators=2*24, criterion='entropy'
    )
    fs_erf_aucs = classifier_crossval_performance(
        X_sel, y.values, classifier=fs_extra_rf, bins=120
    )

    print('Compare GNB to Features Slected GNB')
    welch_test(fs_erf_aucs, erf_aucs)
