from platform import node
from utils import *
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# TODO: use relu as activation func
def relu(z):
    if z >= 0:
        return z
    return 0

def step(z):
    if z >= 0:
        return 1
    return 0

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
    y1 = sigmoid(z1)
    # print("z1 shape: ", z1.shape)
    # print("y1 shape: ", y1.shape)

    z2 = np.dot(y1, weights["W2"].T) + np.tile(weights["B2"].T, (M, 1))
    y2 = sigmoid(z2)
    # print("z2 shape: ", z2.shape)
    # print("y2 shape: ", y2.shape)

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

    y1 = node_output["Y1"]
    y2 = node_output["Y2"]
    w2 = W["W2"]

    dz2 = y2 - t
    dw2 = 1/M * np.dot(dz2.T, y1)
    db2 = 1/M * np.sum(dz2)

    dz1 = np.dot(dz2, w2) * (y1 * (1 - y1))
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
    # print("w1_new shape: ", w1_new.shape)
    # print("b1_new shape: ", b1_new.shape)
    # print("w2_new shape: ", w2_new.shape)
    # print("b2_new shape: ", b2_new.shape)

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

def train(X_train, t_train, X_val, t_val, nn_dims, alpha):
    train_losses = []
    valid_accs = []

    W = init_weights(nn_dims["n_x"], nn_dims["n_h"], nn_dims["n_y"])

    w_best = None
    acc_best = 0
    epoch_best = 0

    # num_batches = int(np.ceil(len(X_train)/batch_size))  
    for epoch in range(MaxEpoch):
        node_output = forward_prop(X_train, W)

        loss = get_cross_entropy_loss(node_output["Y2"], t_train)
        print("loss this epoch: ", loss)

        grads = back_prop(W, X_train, t_train, node_output)
        print("done back prop")

        W = update_weights(W, grads, alpha)
        print("done updating weights")

        # TODO: update weights based on gradients 

        # grad = np.dot(X_batch.T, (y-t_batch))
        # w = w - alpha*grad
        # for batch in range(num_batches):

        #     X_batch = X_train[batch*batch_size: (batch+1)*batch_size]
        #     t_batch = np.concatenate(t_train[batch*batch_size: (batch+1)*batch_size])

        #     forward_prop()
        #     y, t_hat, loss, acc = predict(X_batch, w, t_batch)
        #     loss_this_epoch += loss

        #     grad = np.dot(X_batch.T, (y-t_batch))
        #     w = w - alpha*grad

        # Perform validation on the validation set by accuracy
        # _, _, _, acc = predict(X_val, w, t_val)
        # print("epoch acc: ", acc)

        # # Append training loss and accuracy for the epoch
        # train_losses.append(loss_this_epoch/num_batches)
        # valid_accs.append(acc)

        # Keep track of the best validation epoch, accuracy, and the weights
        # if acc > acc_best:
        #     acc_best = acc
        #     w_best = w
        #     epoch_best = epoch
    
    w_best = W
    return epoch_best, acc_best,  w_best, train_losses, valid_accs


##############################
# Main code starts here

# Single hidden layer with 4 nodes
# TODO: tune this
n_h = 12

alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 100      # Maximum epoch - default 50
decay = 0.          # weight decay

# logistic regression classifier
X_train, t_train, X_val, t_val, X_test, t_test = get_data()

nn_dims = {
    "n_x": X_train.shape[1],
    "n_h": n_h,
    "n_y": 1,
}

epoch_best, acc_best, w_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val, nn_dims, alpha)
print("epoch best: ", epoch_best)
print("acc best: ", acc_best)

t_hat = predict(X_test, w_best)
acc_test = get_accuracy(t_test, t_hat)
print("test acc: ", acc_test)