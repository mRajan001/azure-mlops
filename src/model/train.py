# Import libraries

import argparse
import glob
import os

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlflow

# define functions

def split_data(df):
    # a funstion to split data
    # last column = y
    y = df.iloc[:, -1]
    # all columns except last = X
    X = df.iloc[:, :-1]
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    #print the length of the train and test data
    print(f'Length of X_train: {len(X_train)}')
    print("")
    print(f'Length of X_test: {len(X_test)}')
    print("")
    print(f'Length of y_train: {len(y_train)}')
    print("")
    print(f'Length of y_test: {len(y_test)}')

    return X_train, X_test, y_train, y_test

def main(args):
    # TO DO: enable autologging with MLflow
    mlflow.autolog()
    
    print('Training data path', args.training_data)

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()
    # print(args)
    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
