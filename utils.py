import pandas as pd
from sklearn.utils import shuffle

def get_data():
    data = pd.read_csv('./train.csv')
    data = shuffle(data)
    # print(data)
    # print(data.describe())

    # Fill missing age data using mean
    data["Age"].fillna(data["Age"].mean(), inplace=True)

    # Encode Sex feature into numerical form
    data["Sex_encoded"]=pd.factorize(data["Sex"])[0]

    X_ = data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_encoded"]]
    Y_ = data[["Survived"]]

    X_train = X_[:500]
    Y_train = Y_[:500]

    X_val = X_[500:695]
    Y_val = Y_[500:695]

    X_test = X_[695:]
    Y_test = Y_[695:]

    return X_train.to_numpy(), Y_train.to_numpy(), X_val.to_numpy(), Y_val.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()