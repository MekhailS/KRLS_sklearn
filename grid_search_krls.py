import random

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from krls import KRLS


def generate_data(seed, len_train_data=200, len_test_data=200, range=(-2, 2), noise_std=0.0):
    np.random.seed(seed)

    X_train = np.random.uniform(range[0], range[1], len_train_data)
    y_train = np.sinc(X_train) + np.random.normal(0, noise_std, size=len(X_train))

    X_test = np.random.uniform(range[0], range[1], len_test_data)
    y_test = np.sinc(X_test)

    return X_train, y_train, X_test, y_test


def main(seed):
    parameters = {
        "delta_threshold": list(np.linspace(start=0.1, stop=2, num=20)), # 0.3
        "gamma": list(np.linspace(start=0.1, stop=3, num=40)) # 0.8435897435897436
    }

    X_train, y_train, X_test, y_test = generate_data(seed)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    test_fold = [-1]*len(X_train) + [0]*len(X_test)

    split = PredefinedSplit(test_fold)

    clf = GridSearchCV(KRLS(), parameters, scoring="neg_root_mean_squared_error", cv=split)
    clf.fit(X, y)

    results = pd.DataFrame(clf.cv_results_)
    print({
        "best_params": clf.best_params_,
        "best_score": clf.best_score_
    })
    print()


if __name__ == '__main__':
    seeds = [random.randint(0, 2022) for _ in range(30)]
    for seed in seeds:
        print(f"SEED {seed}")
        main(seed)
