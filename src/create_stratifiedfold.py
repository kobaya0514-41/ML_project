import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("project/input/filename.csv")

    df["kfold"] = -1

    df = df.sample(frac = 1).reset_index(drop = True)

    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits = 5)

    for fold,(trn_,val_) in enumerate(kf.split(X = df ,y = y)):
        df.loc[val_,'kfold'] = fold

        df.to_csv("project/input/adult_folds.csv" , index = False)
