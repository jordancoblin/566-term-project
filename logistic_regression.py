from utils import *
from sklearn.metrics import log_loss
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.special import expit

def sigmoid(z):
    return expit(z)

def one_hot(t, k):
    c = np.concatenate(t)
    t_hot = np.zeros((len(c), k))
    t_hot[np.arange(len(t_hot)),c] = 1
    return t_hot

def predict(X, w, t=None):
    # X: Nsample x (d)
    # w: (d)

    z = np.dot(X, w)
    y = sigmoid(z)

    threshold = 0.5 
    t_hat = np.array(list(map(lambda x: 1 if x >= threshold else 0, y)))

    loss = get_cross_entropy_loss(y, t)
    acc = accuracy_score(t, t_hat)
    return y, t_hat, loss, acc


def train(X_train, t_train, X_val, t_val, params):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    train_losses = []
    valid_accs = []

    batch_size = params['batch_size']
    epochs = params['epochs']
    alpha = params['alpha']

    w = np.zeros(X_train.shape[1])

    w_best = None
    acc_best = 0
    epoch_best = 0

    num_batches = int(np.ceil(len(X_train)/batch_size))  
    for epoch in range(epochs):

        loss_this_epoch = 0
        for batch in range(num_batches):

            X_batch = X_train[batch*batch_size: (batch+1)*batch_size]
            t_batch = np.concatenate(t_train[batch*batch_size: (batch+1)*batch_size])

            y, t_hat, loss, acc = predict(X_batch, w, t_batch)
            loss_this_epoch += loss

            grad = (1/batch_size) * np.dot(X_batch.T, (y-t_batch))
            w = w - alpha*grad

        # Perform validation on the validation set by accuracy
        _, _, _, acc = predict(X_val, w, t_val)

        # Append training loss and accuracy for the epoch
        train_losses.append(loss_this_epoch/num_batches)
        valid_accs.append(acc)

        # Keep track of the best validation epoch, accuracy, and the weights
        if acc > acc_best:
            acc_best = acc
            w_best = w
            epoch_best = epoch
    
    return epoch_best, acc_best,  w_best, train_losses, valid_accs

def tune_params(X_train, t_train, X_val, t_val):
    acc_best = 0
    alpha_best = None
    batch_size_best = None

    alphas = [0.9, 0.1, 0.01, 0.001, 0.0001]
    batch_sizes = [200, 100, 50, 10]

    num_iterations = 20
    for alpha in alphas:
        for batch_size in batch_sizes:
            summed_acc = 0
            for i in range(num_iterations):
                params = {
                    'batch_size': batch_size,
                    'epochs': num_iterations,
                    'alpha': alpha,
                }
                _, acc, _, _, _ = train(X_train, t_train, X_val, t_val, params)
                summed_acc += acc

            avg_acc = summed_acc/num_iterations
            print(f'alpha: {alpha}, batch: {batch_size}, avg_acc: {avg_acc}')

            if avg_acc > acc_best:
                acc_best = avg_acc
                alpha_best = alpha
                batch_size_best = batch_size

    tuned_params = {
        'batch_size': batch_size_best,
        'alpha': alpha_best,
    }
    return acc_best, tuned_params

##############################
# Main code starts here
# alpha = 0.1      # learning rate
# batch_size = 100    # batch size
# MaxEpoch = 1000      # Maximum epoch - default 50
# decay = 0.          # weight decay

# logistic regression classifier
X_train, t_train, X_val, t_val, X_test, t_test = get_data()

# Tune hyperparameters
acc, params = tune_params(X_train, t_train, X_val, t_val)
alpha = params['alpha']
batch_size = params['batch_size'] 
print(f'TUNED: alpha: {alpha}, batch: {batch_size}, acc: {acc}')

# Train using tuned hyperparams and more Epochs
MaxEpoch = 1000      # Maximum epoch

params = {
    'batch_size': batch_size,
    'epochs': MaxEpoch,
    'alpha': alpha,
}
epoch_best, acc_best, w_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val, params)
print("epoch best: ", epoch_best)
print("acc best: ", acc_best)

_, _, _, acc_test = predict(X_test, w_best, t_test)
print("test acc: ", acc_test)

plt.figure()
plt.title("Logistic Regression Training Loss over Epochs")
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.plot(train_losses)
plt.savefig('lr_training_loss' + '.png')

plt.figure()
plt.title("Logisting Regression Validation Accuracy over Epochs")
plt.xlabel('epoch')
plt.ylabel('validation accuracy')
plt.plot(valid_accs)
plt.savefig('lr_validation_accuracy' + '.png')

