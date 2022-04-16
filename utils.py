import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def get_cross_entropy_loss(y, t):
    epsilon = 1e-5    
    return -np.average(t * np.log(y + epsilon) + (1 - t) * np.log(1 - y + epsilon))

def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    print("t shape: ", t.shape)
    print("t_hat shape: ", t_hat.shape)
    num_correct = 0
    for i, t_m in enumerate(t):
        if t_m == t_hat[i]:
            num_correct += 1
    return num_correct/len(t)

def get_data():
    data = pd.read_csv('./train.csv')
    data = shuffle(data)
    # print(data)
    # print(data.describe())

    # Fill missing age data using mean
    data["Age"].fillna(data["Age"].mean(), inplace=True)

    # Encode Sex feature into numerical form
    data["Sex_encoded"]=pd.factorize(data["Sex"])[0]

    X_ = data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_encoded"]].to_numpy()
    Y_ = data[["Survived"]].to_numpy()

    # X_aug = np.concatenate([np.ones([len(X_), 1]), X_], axis=1)
    X_aug = X_

    # X_aug = np.concatenate(
    #     (np.ones([data.shape[0], 1]), data), axis=1)

    X_train = X_aug[:500]
    Y_train = Y_[:500]

    X_val = X_aug[500:695]
    Y_val = Y_[500:695]

    X_test = X_aug[695:]
    Y_test = Y_[695:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test