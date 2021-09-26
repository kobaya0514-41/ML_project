from numpy.lib.function_base import percentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

class UnivariateFeatureSelction:
    def __init__(self, n_features, problem_type, scoring):
        """
        :param n_features: if float type is SelectPercentile else SelectKBest
        :param problem_type:classification or regression
        :param scoring: if problem_type is classification f_classif,chi2,mutual_info_classif else f_regression,mutual_info_regression
        """
        if problem_type == "classification":
            valid_scoring = {
                "f_classif":f_classif,
                "chi2":chi2,
                "mutual_info_classif":mutual_info_classif
            }
        else:
            valid_scoring = {
                "f_regression":f_regression,
                "mutual_info_regression":mutual_info_regression
            }

        #exception
        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")

        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile = int(n_features * 100)
            )
        else:
            raise Exception("Invalid type of feature")
        
        #fit function
        def fit(self, X, y):
            return self.selection.fit(X, y)

        #transform function
        def transform(self, X):
            return self.selection.transform(X)
        
        #fit_transform function
        def fit_transform(self, X, y):
            return self.selection.fit_transform(X, y)
