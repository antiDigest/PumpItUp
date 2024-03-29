import xgboost

from sklearn.metrics import classification_report
import sys
from sklearn.model_selection import ShuffleSplit


def xgb(xtrain, ytrain, xtest, ytest):
    """
        Gradient Boosted Trees:

        from XGBOOST library
        produces classification report for both train and test data

    """

    print("Running XGBoost")
    x = xgboost.XGBClassifier(max_depth=18, learning_rate=0.03,
                              n_estimators=400, gamma=3, nthread=4)
    sys.stdout.flush()

    # print("Plotting Learning curve")
    # sys.stdout.flush()
    # cv = ShuffleSplit(n_splits=15, test_size=0.2, random_state=0)
    # plot_learning_curve(x, "GBTree Classifier Training", xtrain, ytrain,
    #                     ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    x.fit(xtrain, ytrain)

    pred = x.predict(xtrain)
    print(classification_report(ytrain, pred))

    pred = x.predict(xtest)
    print(classification_report(ytest, pred))

    return pred
