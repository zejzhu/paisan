import argparse
import numpy as np
import pandas as pd
import timeit
from compareFeat import eval_model, evaluate_unreg
from sklearn.metrics import mean_squared_error, r2_score

# part of solution
from sklearn.linear_model import SGDRegressor


def evaluate_sgd(trainDF, testDF, mepoch):
    """
    Evaluate the performance of a SGD-based linear regression
    without regularization. Train the regression using the 
    training dataset and evaluate the performance on the test set.

    Parameters
    ----------
    trainDF : pandas.DataFrame
        the training dataframe
    testDF : pandas.DataFrame
        the test dataframe   
    mepoch : int
        Maximum number of epochs

    Returns
    -------
    res : dictionary
        return the dictionary with the following keys --
        r2, mse, time, nfeat
    """
    xtrain = trainDF.iloc[:, :-1]
    ytrain = trainDF.iloc[:, -1]
    xtest = testDF.iloc[:, :-1]
    ytest = testDF.iloc[:, -1]

    start = timeit.default_timer()
    model = SGDRegressor(max_iter = mepoch)
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)

    r2 = r2_score(ytest, ypred)
    mse = mean_squared_error(ytest, ypred)
    nfeat = int(np.sum(model.coef_ != 0))



    time_lr = timeit.default_timer() - start
    return {
        'r2': r2,
        'mse': mse,
        'time': time_lr,
        'nfeat': nfeat
    }


def main():
    """
    Main file to run from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("fullTrain",
                        help="filename of the full-featured training data")
    parser.add_argument("fullTest",
                        help="filename of the full-featured  test data")

    args = parser.parse_args()
    # load the data
    print("Loading data ----")
    full_train = pd.read_csv(args.fullTrain)
    full_test = pd.read_csv(args.fullTest)
    print("Training models now ----")
    perf = {}
    print("Closed Form")
    perf["closed"] = evaluate_unreg(full_train, full_test)
    print("Max Epoch = 1")
    perf["sgd (e=1) r1"] = evaluate_sgd(full_train, full_test, 1)
    perf["sgd (e=1) r2"] = evaluate_sgd(full_train, full_test, 1)
    perf["sgd (e=1) r3"] = evaluate_sgd(full_train, full_test, 1)
    print("Max Epoch = 5")
    perf["sgd (e=5) r1"] = evaluate_sgd(full_train, full_test, 5)
    perf["sgd (e=5) r2"] = evaluate_sgd(full_train, full_test, 5)
    perf["sgd (e=5) r3"] = evaluate_sgd(full_train, full_test, 5)
    print("Max Epoch = 25")
    perf["sgd (e=25) r1"] = evaluate_sgd(full_train, full_test, 25)
    perf["sgd (e=25) r2"] = evaluate_sgd(full_train, full_test, 25)
    perf["sgd (e=25) r3"] = evaluate_sgd(full_train, full_test, 25)

    print(pd.DataFrame.from_dict(perf, orient='index'))


if __name__ == "__main__":
    main()



