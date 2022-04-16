from utils import *
import numpy as np

def relu(z):
    if z >= 0:
        return z
    return 0

def init_weights(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(n_h)

    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(n_y)

    weights = {
        "W1": w1,
        "B1": b1,
        "W2": w2,
        "B2": b2,
    }
    return weights

def forward_prop(X, weights):
    z1 = np.dot(weights["W1"], X) + weights["B1"]
    y1 = relu(z1)

    z2 = np.dot(weights["W2"], X) + weights["B2"]
    y2 = relu(z2)

    node_output = {
        "Z1" : z1,
        "Y1": y1,
        "Z2": z2,
        "Y2": y2,
    }
    return node_output

def back_prop():
    return

def predict():
    return

def train(X_train, t_train, X_val, t_val, nn_dims):
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

        grads = back_prop(W, node_output)

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
n_h = 4

alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 10      # Maximum epoch - default 50
decay = 0.          # weight decay

# logistic regression classifier
X_train, t_train, X_val, t_val, X_test, t_test = get_data()

nn_dims = {
    "n_x": X_train.shape[1],
    "n_h": n_h,
    "n_y": 1,
}

epoch_best, acc_best, w_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val, nn_dims)
print("epoch best: ", epoch_best)
print("acc best: ", acc_best)

_, _, _, acc_test = predict(X_test, w_best, t_test)
print("test acc: ", acc_test)