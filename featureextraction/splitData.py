import argparse
import pandas as pd
import sklearn.model_selection as model_selection

def create_split(df):
    """
    Create the train-test split. The method should be 
    randomized so each call will likely yield different 
    results.
    
    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    train_df : pandas.DataFrame
        return the training dataset as a pandas dataframe
    test_df : pandas.DataFrame
        return the test dataset as a pandas dataframe.
    """
    #can use scikit for all
    #so it shall be randomized
    #70 30 split
    train_df, test_df = model_selection.train_test_split(df, test_size=0.3, shuffle=True)
    return train_df, test_df


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("input", 
                        help="filename of training data")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    train_df, test_df = create_split(df)
    #have to name the output file based on the input file
    train_df.to_csv(args.input.replace(".csv", "Train.csv"), index = False)
    test_df.to_csv(args.input.replace(".csv", "Test.csv"), index = False)

    print(args.input + " train shape " + str(train_df.shape))
    print(args.input + " test shape " + str(test_df.shape))




if __name__ == "__main__":
    main()
