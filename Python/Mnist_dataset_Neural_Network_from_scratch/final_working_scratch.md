
# Explanation of below code

+ Below code was almost exclusively taken from https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/ 
    + This script ran  faster than the network2 script.  It was therefore easier to tune parameters with
    + The author modeled his script after the Coursera Andrew Ng deep learning class 
        + https://www.coursera.org/learn/deep-neural-network/lecture/RwqYe/weight-initialization-for-deep-networks
+ Some Changes I made to the script
    + Added in a learning rate decay parameter
    + Tuned the learning rate parameter and amount of neurons in hidden layer
        + I used a coarse to fine search method, but wihtout a grid, it was more trial and error.
    + Added comments after code blocks in order to show I understood the code, but it isn't my code and credit for the code goes to that tutorial above.
        


## Import dataset from sklearn
+ normalize pixel values
+ Create train/test split


```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target

# Scale the numerical bits
X = X/255


# creates an identity matrix with 1's where value exists and returns those values
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)


# Create train test split
# split, reshape, shuffle
samples = 60000
m_test = X.shape[0] - samples
X_train, X_test = X[:samples].T, X[samples:].T
Y_train, Y_test = Y_new[:, :samples], Y_new[:, samples:]
shuffle_index = np.random.permutation(samples)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]
```


```python
def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s


def compute_loss(Y, Y_hat):
    # Multiclass Cross entropy loss function
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L


def feed_forward(X, params):
    cache = {}
    ## sigmoid activation
    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    cache["A1"] = sigmoid(cache["Z1"])
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
    ##  softmax output activation
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

    return cache


def back_propagate(X, Y, params, cache):

    dZ2 = cache["A2"] - Y
    dW2 = (1./m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    dW1 = (1./m_batch) * np.matmul(dZ1, X.T)
    db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


np.random.seed(138)

# hyperparameters
n_x = X_train.shape[0]
neurons = 185
my_learning_rate = 4.5
beta = .9
batch_size = 128
batches = -(-samples // batch_size)
learning_rate_decay = .02

# initialization set
# weights are divded by sqrt(n)
params = {"W1": np.random.randn(neurons, n_x) * np.sqrt(1. / n_x),
          "b1": np.zeros((neurons, 1)) * np.sqrt(1. / n_x),
          "W2": np.random.randn(digits, neurons) * np.sqrt(1. / neurons),
          "b2": np.zeros((digits, 1)) * np.sqrt(1. / neurons)}

V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)


for i in range(17):

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    # momentum based mini batch-SGD
    for j in range(batches):
        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = feed_forward(X, params)
        grads = back_propagate(X, Y, params, cache)
        #add in momentum
        V_dW1 = (beta * V_dW1 + (1. - beta) * grads["dW1"])
        V_db1 = (beta * V_db1 + (1. - beta) * grads["db1"])
        V_dW2 = (beta * V_dW2 + (1. - beta) * grads["dW2"])
        V_db2 = (beta * V_db2 + (1. - beta) * grads["db2"])

        params["W1"] = params["W1"] - my_learning_rate * V_dW1
        params["b1"] = params["b1"] - my_learning_rate * V_db1
        params["W2"] = params["W2"] - my_learning_rate * V_dW2
        params["b2"] = params["b2"] - my_learning_rate * V_db2 
        
    # Add in learning rate decay
    my_learning_rate = max(
        (my_learning_rate / (1 + learning_rate_decay*i)), .02)
    print(learning_rate)
    # update finished epoch paramters and print
    cache = feed_forward(X_train, params)
    train_cost = compute_loss(Y_train, cache["A2"])
    cache = feed_forward(X_test, params)
    test_cost = compute_loss(Y_test, cache["A2"])
    predictions = np.argmax(cache["A2"], axis=0)
    labels = np.argmax(Y_test, axis=0)
    accurate_predictions = sum(predictions == labels)
    print("Epoch {}:Learning rate = {}, accurate predictions = {}, test cost = {}".format(i+1,
                                                                                          my_learning_rate, accurate_predictions, test_cost))

print("Done.")
```

    6
    Epoch 1:Learning rate = 4.5, accurate predictions = 9581, test cost = 0.14064052472400096
    6
    Epoch 2:Learning rate = 4.411764705882353, accurate predictions = 9662, test cost = 0.1042637181555412
    6
    Epoch 3:Learning rate = 4.242081447963801, accurate predictions = 9717, test cost = 0.08609054388595846
    6
    Epoch 4:Learning rate = 4.001963630154529, accurate predictions = 9766, test cost = 0.07306203876945543
    6
    Epoch 5:Learning rate = 3.705521879772712, accurate predictions = 9778, test cost = 0.06926331587349448
    6
    Epoch 6:Learning rate = 3.3686562543388288, accurate predictions = 9790, test cost = 0.0661912522686922
    6
    Epoch 7:Learning rate = 3.007728798516811, accurate predictions = 9800, test cost = 0.06406923670334039
    6
    Epoch 8:Learning rate = 2.638358595190185, accurate predictions = 9798, test cost = 0.06286531677517795
    6
    Epoch 9:Learning rate = 2.274447064819125, accurate predictions = 9817, test cost = 0.06198916368807139
    6
    Epoch 10:Learning rate = 1.9274975125585807, accurate predictions = 9802, test cost = 0.062046180428826085
    6
    Epoch 11:Learning rate = 1.6062479271321506, accurate predictions = 9819, test cost = 0.059927406306813785
    6
    Epoch 12:Learning rate = 1.3165966615837301, accurate predictions = 9819, test cost = 0.05909621913605408
    6
    Epoch 13:Learning rate = 1.0617715012772018, accurate predictions = 9820, test cost = 0.06032692423923074
    6
    Epoch 14:Learning rate = 0.8426757946644459, accurate predictions = 9815, test cost = 0.060079622151832705
    6
    Epoch 15:Learning rate = 0.6583404645815983, accurate predictions = 9818, test cost = 0.05958296724681382
    6
    Epoch 16:Learning rate = 0.5064157419858448, accurate predictions = 9819, test cost = 0.06003421681048306
    6
    Epoch 17:Learning rate = 0.38364828938321577, accurate predictions = 9821, test cost = 0.059590976087357035
    Done.
    

# Results 
+ The original script from the author, reduced the cost function on training set to about 0.0802 in 9 epochs
+ Tuning has allowed us to reduce test cost to .0595 in 17 epochs
+ below I print the classification report


```python
cache = feed_forward(X_test, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)

print(classification_report(predictions, labels,digits=4))


```

                 precision    recall  f1-score   support
    
              0     0.9888    0.9858    0.9873       983
              1     0.9903    0.9912    0.9907      1134
              2     0.9816    0.9797    0.9806      1034
              3     0.9861    0.9794    0.9827      1017
              4     0.9776    0.9796    0.9786       980
              5     0.9731    0.9875    0.9802       879
              6     0.9823    0.9802    0.9812       960
              7     0.9835    0.9825    0.9830      1029
              8     0.9784    0.9784    0.9784       974
              9     0.9772    0.9762    0.9767      1010
    
    avg / total     0.9821    0.9821    0.9821     10000
    
    
