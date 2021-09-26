import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    path = "C:/Users/youichi_kobayashi/python/project/input/mnist_train.csv"
    df = pd.read_csv(path,dtype='int8',header=None)

    df["kfold"] = -1

    df = df.sample(frac = 1).reset_index(drop = True)
    kf = model_selection.KFold(n_splits = 5)

    for fold,(trn_,val_) in enumerate(kf.split(X = df)):
        df.loc[val_,'kfold'] = fold

        df.to_csv("project/input/train_fold.csv" , index = False)
