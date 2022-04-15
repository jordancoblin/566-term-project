from utils import get_data
from sklearn.metrics import log_loss
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

# def softmax(z):
#     max = np.max(z,axis=1,keepdims=True)
#     e_z = np.exp(z - max)
#     sum = np.sum(e_z, axis=1, keepdims=True)
#     return e_z/sum

def sigmoid(z):
    # print("z: ", z)
    return 1 / (1 + np.exp(-z))

    # return np.array(list(map(lambda x: 1/(1+math.exp(-x)), z)))

def one_hot(t, k):
    c = np.concatenate(t)
    t_hot = np.zeros((len(c), k))
    t_hot[np.arange(len(t_hot)),c] = 1
    return t_hot

def cross_entropy(y, t):
    epsilon = 1e-5    
    return -np.average(t * np.log(y + epsilon) + (1 - t) * np.log(1 - y + epsilon))

def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    num_correct = 0
    for i, t_m in enumerate(t):
        if t_m == t_hat[i]:
            num_correct += 1
    return num_correct/len(t)

def predict(X, w, t=None):
    # X: Nsample x (d)
    # w: (d)

    z = np.dot(X, w)
    y = sigmoid(z)

    threshold = 0.5 
    t_hat = np.array(list(map(lambda x: 1 if x >= threshold else 0, y)))

    loss = cross_entropy(y, t)
    acc = get_accuracy(t, t_hat)
    return y, t_hat, loss, acc


def train(X_train, t_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    train_losses = []
    valid_accs = []

    w = np.zeros(X_train.shape[1])

    w_best = None
    acc_best = 0
    epoch_best = 0

    num_batches = int(np.ceil(len(X_train)/batch_size))  
    for epoch in range(MaxEpoch):

        loss_this_epoch = 0
        for batch in range(num_batches):

            X_batch = X_train[batch*batch_size: (batch+1)*batch_size]
            t_batch = np.concatenate(t_train[batch*batch_size: (batch+1)*batch_size])

            y, t_hat, loss, acc = predict(X_batch, w, t_batch)
            loss_this_epoch += loss

            grad = np.dot(X_batch.T, (y-t_batch))
            w = w - alpha*grad

        # Perform validation on the validation set by accuracy
        _, _, _, acc = predict(X_val, w, t_val)
        print("epoch acc: ", acc)

        # Append training loss and accuracy for the epoch
        train_losses.append(loss_this_epoch/num_batches)
        valid_accs.append(acc)

        # Keep track of the best validation epoch, accuracy, and the weights
        if acc > acc_best:
            acc_best = acc
            w_best = w
            epoch_best = epoch
    
    return epoch_best, acc_best,  w_best, train_losses, valid_accs


##############################
# Main code starts here
alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 1000      # Maximum epoch - default 50
decay = 0.          # weight decay

# logistic regression classifier
X_train, t_train, X_val, t_val, X_test, t_test = get_data()
print(X_train)

epoch_best, acc_best, w_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val)
print("epoch best: ", epoch_best)
print("acc best: ", acc_best)

_, _, _, acc_test = predict(X_test, w_best, t_test)
print("test acc: ", acc_test)

# plot_data(X, t, w, b, is_logistic=True,
#           figure_name='dataset_A_logistic.png')

plt.figure()
plt.title("Training Loss over Epochs")
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.plot(train_losses)
plt.savefig('training_loss' + '.png')

plt.figure()
plt.title("Validation Accuracy over Epochs")
plt.xlabel('epoch')
plt.ylabel('validation accuracy')
plt.plot(valid_accs)
plt.savefig('validation_accuracy' + '.png')

