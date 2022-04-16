from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from utils import *

##############################
# Main code starts here

X_train, t_train, X_val, t_val, X_test, t_test = get_data()

# Use validation set as well for training
X_train = np.concatenate((X_train, X_val))
t_train = np.concatenate((t_train, t_val))

# Tune parameters using GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'sigmoid']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train, t_train.ravel())

print(grid.best_params_)

C = grid.best_params_["C"]
kernel = grid.best_params_["kernel"]
gamma = grid.best_params_['gamma']

# Best params were found to be:
# C=1000, kernel='rbf', gamma='scale'

# Fit SVM with tuned params
classifier = SVC(C=C, kernel=kernel, gamma=gamma)
classifier.fit(X_train, t_train)

t_hat = classifier.predict(X_test)
acc = accuracy_score(t_test, t_hat)
print("test acc: ", acc)
