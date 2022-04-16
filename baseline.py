from utils import *
from sklearn.metrics import accuracy_score

# Use Zero Rule baseline - i.e. predict not survived for each passenger
def predict(X):
    return np.zeros(len(X))

##############################
# Main code starts here

_, _, _, _, X_test, t_test = get_data()

t_hat = predict(X_test)
acc = accuracy_score(t_test, t_hat)
print("test acc: ", acc)