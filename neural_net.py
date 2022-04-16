from platform import node
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# Derivative of ReLU function
def step(z):
    return np.heaviside(z, 0)

def init_weights(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))

    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    weights = {
        "W1": w1,
        "B1": b1,
        "W2": w2,
        "B2": b2,
    }
    return weights

def forward_prop(X, weights):
    # W: Ny x Nx
    # X: M x Nx
    M = X.shape[0]
    z1 = np.dot(X, weights["W1"].T) + np.tile(weights["B1"].T, (M, 1))
    # y1 = sigmoid(z1)
    y1 = relu(z1)

    z2 = np.dot(y1, weights["W2"].T) + np.tile(weights["B2"].T, (M, 1))
    y2 = sigmoid(z2)

    node_output = {
        "Z1" : z1,
        "Y1": y1,
        "Z2": z2,
        "Y2": y2,
    }
    return node_output

def back_prop(W, X, t, node_output):
    # W: Ny x Nx
    # X: M x Nx
    # t: M x 1
    M = len(X)

    z1 = node_output["Z1"]
    y1 = node_output["Y1"]
    y2 = node_output["Y2"]
    w2 = W["W2"]

    dz2 = y2 - t
    dw2 = 1/M * np.dot(dz2.T, y1)
    db2 = 1/M * np.sum(dz2)

    # dz1 = np.dot(dz2, w2) * (y1 * (1 - y1))
    dz1 = np.dot(dz2, w2) * step(z1)
    dw1 = 1/M * np.dot(dz1.T, X)
    db1 = 1/M * np.sum(dz1)

    grads = {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }
    return grads

def update_weights(W, grads, alpha):
    w1_new = W["W1"] - alpha * grads["dw1"]
    b1_new = W["B1"] - alpha * grads["db1"]
    w2_new = W["W2"] - alpha * grads["dw2"]
    b2_new = W["B2"] - alpha * grads["db2"]
    
    weights_new = {
        "W1": w1_new,
        "B1": b1_new,
        "W2": w2_new,
        "B2": b2_new,
    }
    return weights_new

def predict(X, W):
    node_output = forward_prop(X, W)
    threshold = 0.5
    return node_output["Y2"] > threshold

def train(X_train, t_train, X_val, t_val, nn_dims, params):
    train_losses = []
    valid_accs = []

    batch_size = params['batch_size']
    epochs = params['epochs']
    alpha = params['alpha']

    W = init_weights(nn_dims["n_x"], nn_dims["n_h"], nn_dims["n_y"])

    w_best = None
    acc_best = 0
    epoch_best = 0

    num_batches = int(np.ceil(len(X_train)/batch_size))  
    for epoch in range(epochs):

        loss_this_epoch = 0
        for batch in range(num_batches):
            X_batch = X_train[batch*batch_size: (batch+1)*batch_size]
            t_batch = t_train[batch*batch_size: (batch+1)*batch_size]

            node_output = forward_prop(X_batch, W)

            loss = get_cross_entropy_loss(node_output["Y2"], t_batch)
            loss_this_epoch += loss

            grads = back_prop(W, X_batch, t_batch, node_output)

            W = update_weights(W, grads, alpha)


        # Perform validation on the validation set by accuracy
        t_hat = predict(X_val, W)
        acc = accuracy_score(t_val, t_hat)

        # Append training loss and accuracy for the epoch
        train_losses.append(loss_this_epoch/num_batches)
        valid_accs.append(acc)

        if acc > acc_best:
            acc_best = acc
            w_best = W
            epoch_best = epoch
    
    return epoch_best, acc_best,  w_best, train_losses, valid_accs

def tune_params(X_train, t_train, X_val, t_val):
    acc_best = 0
    alpha_best = None
    batch_size_best = None
    n_h_best = None

    alphas = [0.9, 0.1, 0.01, 0.001, 0.0001]
    batch_sizes = [200, 100, 50, 10]
    n_hs = [4, 8, 16, 32, 64]

    num_iterations = 10
    for alpha in alphas:
        for batch_size in batch_sizes:
            for n_h in n_hs:
                summed_acc = 0
                for i in range(num_iterations):
                    nn_dims = {
                        "n_x": X_train.shape[1],
                        "n_h": n_h,
                        "n_y": 1,
                    }
                    params = {
                        'batch_size': batch_size,
                        'epochs': num_iterations,
                        'alpha': alpha,
                    }
                    _, acc, _, _, _ = train(X_train, t_train, X_val, t_val, nn_dims, params)
                    summed_acc += acc

                avg_acc = summed_acc/num_iterations
                print(f'alpha: {alpha}, batch: {batch_size}, n_h: {n_h}, avg_acc: {avg_acc}')

                if avg_acc > acc_best:
                    acc_best = avg_acc
                    alpha_best = alpha
                    batch_size_best = batch_size
                    n_h_best = n_h

    tuned_params = {
        'batch_size': batch_size_best,
        'n_h': n_h_best,
        'alpha': alpha_best,
    }
    return acc_best, tuned_params

##############################
# Main code starts here

X_train, t_train, X_val, t_val, X_test, t_test = get_data()

# Tune hyperparameters
acc, params = tune_params(X_train, t_train, X_val, t_val)
n_h = params['n_h']
alpha = params['alpha']
batch_size = params['batch_size'] 
print(f'TUNED: alpha: {alpha}, batch: {batch_size}, n_h: {n_h}, acc: {acc}')

# Train using tuned hyperparams and more Epochs
MaxEpoch = 1000      # Maximum epoch

nn_dims = {
    "n_x": X_train.shape[1],
    "n_h": n_h,
    "n_y": 1,
}
params = {
    'batch_size': batch_size,
    'epochs': MaxEpoch,
    'alpha': alpha,
}
epoch_best, acc_best, w_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val, nn_dims, params)
print("epoch best: ", epoch_best)
print("acc best: ", acc_best)

t_hat = predict(X_test, w_best)
acc_test = accuracy_score(t_test, t_hat)
print("test acc: ", acc_test)

plt.figure()
plt.title("Neural Network Training Loss over Epochs")
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.plot(train_losses)
plt.savefig('nn_training_loss' + '.png')

plt.figure()
plt.title("Neural Network Validation Accuracy over Epochs")
plt.xlabel('epoch')
plt.ylabel('validation accuracy')
plt.plot(valid_accs)
plt.savefig('nn_validation_accuracy' + '.png')