import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def get_data():
    data = pd.read_csv('./train.csv')
    # data = shuffle(data)
    # print(data)
    # print(data.describe())

    # Fill missing age data using mean
    data["Age"].fillna(data["Age"].mean(), inplace=True)

    # Encode Sex feature into numerical form
    data["Sex_encoded"]=pd.factorize(data["Sex"])[0]

    X_ = data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_encoded"]].to_numpy()
    Y_ = data[["Survived"]].to_numpy()

    X_aug = np.concatenate([np.ones([len(X_), 1]), X_], axis=1)

    # X_aug = np.concatenate(
    #     (np.ones([data.shape[0], 1]), data), axis=1)

    X_train = X_aug[:500]
    Y_train = Y_[:500]

    X_val = X_aug[500:695]
    Y_val = Y_[500:695]

    X_test = X_aug[695:]
    Y_test = Y_[695:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test