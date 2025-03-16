import argparse
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

def cal_corr(df):
    """
    Compute the Pearson correlation matrix
    
    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    corrDF : pandas.DataFrame
        The correlation between the different columns
    """
    corrDF = df.corr(method='pearson')
    return corrDF


def select_features(trainDF, testDF):
    """
    Preprocess the features
    
    Parameters
    ----------
    trainDF : pandas.DataFrame
        the training dataframe
    testDF : pandas.DataFrame
        the test dataframe

    Returns
    -------
    trainDF : pandas.DataFrame
        return the feature-selected trainDF dataframe
    testDF : pandas.DataFrame
        return the feature-selected testDT dataframe
    """
    #target is always last columns
    target = trainDF.columns[-1]
    #drop features with absolute correlation to target below 0.1
    #apply same transforms to train and test
    corrDF = cal_corr(trainDF)
    targetcorr = corrDF[target].abs()
    trainDF.drop(columns=targetcorr[targetcorr < 0.1].index, inplace = True)
    testDF.drop(columns=targetcorr[targetcorr < 0.1].index, inplace = True)

    #drop features with abs correlation to non target featuers above 0.8
    #drop the one that is least strongly correlated to target
    corrDF = cal_corr(trainDF)
    #find pairs of features
    pairs = set()
    for i in range(len(corrDF.columns)):
        for j in range(i + 1, len(corrDF.columns)):
            if abs(corrDF.iloc[i, j]) > 0.7: #value at index
                pairs.add((corrDF.columns[i], corrDF.columns[j]))

    #find feature thats least correlated to target adn drop it
    for (f1, f2) in pairs:
        #SKIP IF ONE OF THEM HAS ALREADY BEEN REMOVED
        if not (f1 in trainDF.columns and f2 in trainDF.columns):
            continue
        if targetcorr[f1] < targetcorr[f2]:
            trainDF.drop(columns=f1, inplace = True)
            testDF.drop(columns=f1, inplace = True)
        else:
            trainDF.drop(columns=f2, inplace = True)
            testDF.drop(columns=f2, inplace = True)


    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("inTrain",
                        help="filename of the training data")
    parser.add_argument("inTest",
                        help="filename of the test data")
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")

    args = parser.parse_args()
    # load the train and test data
    train_df = pd.read_csv(args.inTrain)
    test_df = pd.read_csv(args.inTest)


    print("Original Training Shape:", train_df.shape)
    # calculate the training correlation
    train_df, test_df = select_features(train_df,
                                        test_df)
    print("Transformed Training Shape:", train_df.shape)
    # save it to csv
    train_df.to_csv(args.outTrain, index=False)
    test_df.to_csv(args.outTest, index=False)
    
    """
    #calc correlation matrix
    #plot
    #tfidf
    corrDF = cal_corr(train_df)
    sns.heatmap(corrDF)
    plt.title('TFIDF Correlation Matrix')
    plt.show()
    """
    
    """
    #calc correlation matrix
    #plot
    #binary
    corrDF = cal_corr(train_df)
    sns.heatmap(corrDF)
    plt.title('Binary Correlation Matrix')
    plt.show()
    """

if __name__ == "__main__":
    main()



