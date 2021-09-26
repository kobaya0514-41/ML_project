from os import path
import argparse

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

import model_dispatcher

def run(fold, model):
    path = "C:/Users/youichi_kobayashi/python/project/input/train_fold.csv"
    df = pd.read_csv(path,dtype='int8')

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop("0",axis=1).values
    y_train = df_train['0'].values

    x_valid = df_valid.drop("0",axis=1).values
    y_valid = df_valid['0'].values

    clf = model_dispatcher.models[model]
    clf.fit(x_train,y_train)

    preds = clf.predict(x_valid)

    accuracy = metrics.accuracy_score(y_valid,preds)
    print(f"Fold={fold},Accuracy={accuracy}")

    joblib.dump(clf,f"project/models/dt_{fold}.bin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()

    run(
        fold=args.fold,
        model=args.model
    )
