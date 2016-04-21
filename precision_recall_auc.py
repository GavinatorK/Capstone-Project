import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve


def recall_precision_auc(observed, predicted, fname=None):
    '''
    Evaluate the precision-recall curve for a classifier and return tuple of
    precision, recall, and auc. Optional plot.
    Arguments:
    ----------
        - observed: array of observed classes
        - predicted: array of predicted classes
        - fname: path for writing plot to file, writes to file even where plot=False
    Values:
    -------
        - (precision, recall, auc)
    '''
    precision, recall, thresholds = precision_recall_curve(observed, predicted)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, linestyle='-', color='k')
    plt.xlabel('Recall', labelpad=11); plt.ylabel('Precision', labelpad=11)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

    return recall, precision, pr_auc


if __name__ == "__main__":

    # fake data for classification problem
    X, y = make_classification(n_samples=10000, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)

    # random forest classifier
    forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    forest.fit(X_train, y_train)

    # precision-recall plot
    preds = forest.predict_proba(X_test)[:, 1]
    recall, precision, pr_auc = recall_precision_auc(y_test, preds, fname='images/precision_recall_curve.png')
