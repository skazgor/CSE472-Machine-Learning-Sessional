# -*- coding: utf-8 -*-
"""FNN_offline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JrKQJTexZ_3RtDSc3Td3XlH5uk-AYqgx
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import pickle

import torchvision.datasets as ds
import torchvision.transforms as transforms
train_validation_dataset = ds.EMNIST(root='./data', split='letters',
    train=True,
    transform=transforms.ToTensor(),
    download=True)
independent_test_dataset = ds.EMNIST(root='./data',
    split='letters',
    train=False,
    transform=transforms.ToTensor())

print(train_validation_dataset)

data , level=train_validation_dataset[0]
print(data.shape)

train_validation_data = []
train_validation_labels = []

for data, label in train_validation_dataset:
    data_flattened = data.view(-1)
    train_validation_data.append(data_flattened.numpy())
    train_validation_labels.append(label)

train_validation_data = np.array(train_validation_data)
train_validation_labels = np.array(train_validation_labels)

independent_test_data = []
independent_test_labels = []

for data, label in independent_test_dataset:
    data_flattened = data.view(-1)
    independent_test_data.append(data_flattened.numpy())
    independent_test_labels.append(label)

independent_test_data = np.array(independent_test_data)
independent_test_labels = np.array(independent_test_labels)

train_validation_data.shape

independent_test_data.shape

train_validation_labels=train_validation_labels.reshape(-1,1)
independent_test_labels = independent_test_labels.reshape(-1,1)

print(train_validation_labels.shape)
print(independent_test_labels.shape)

encoder = OneHotEncoder(sparse_output=False)

train_validation_labels_onehot = encoder.fit_transform(train_validation_labels)

independent_test_labels_onehot = encoder.transform(independent_test_labels)

print(train_validation_labels_onehot[0])
print(independent_test_labels_onehot[0])

x_test=independent_test_data
y_test=independent_test_labels_onehot

validation_size=.2

x_train, x_validation, y_train , y_validation = train_test_split(
    train_validation_data,
    train_validation_labels_onehot,
    test_size=validation_size,
    random_state=42
)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.is_dropout=None
        self.dropout_probability=.3

    def forward(self, input):
        # forward output from the input
        pass

    def backward(self, output_gradient, learning_rate):
        print('should it be called?')
        # update its parameter and return values for the previous level
        pass

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input.T))

class Sigmoid(Activation):
    def sigmoid(self, x):
        result = np.zeros_like(x, dtype=float)
        mask_pos = (x >= 0)
        mask_neg = ~mask_pos
        result[mask_pos] = 1 / (1 + np.exp(-x[mask_pos]))
        exp_x_neg = np.exp(x[mask_neg])
        result[mask_neg] = exp_x_neg / (1 + exp_x_neg)
        return result

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)


class Relu(Activation):
    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return np.where(x < 0, 0, 1)

    def __init__(self):
        super().__init__(self.relu, self.relu_prime)

class Dense(Layer):
    def __init__(self, input_size, output_size, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.weights = np.random.randn(output_size, input_size)*np.sqrt(2.0 / (input_size + output_size))
        self.bias = np.random.randn(output_size, 1)*np.sqrt(2.0 / (input_size + output_size))

        # Adam optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize Adam optimizer state
        self.m_dw = np.zeros_like(self.weights)
        self.v_dw = np.zeros_like(self.weights)
        self.m_db = np.zeros_like(self.bias)
        self.v_db = np.zeros_like(self.bias)
        self.t = 1


    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weights.T) + self.bias.T


    def backward(self, output_gradient, learning_rate ):

        batch_size , i = self.input.shape

        dw = output_gradient @ self.input
        dw /= batch_size

        db = np.sum(output_gradient, axis=1 , keepdims = True)/batch_size

        dx = self.weights.T @ output_gradient

        # Update Adam optimizer state
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db ** 2)

        # Bias-corrected estimates
        m_dw_hat = self.m_dw / (1 - self.beta1 ** self.t)
        v_dw_hat = self.v_dw / (1 - self.beta2 ** self.t)
        m_db_hat = self.m_db / (1 - self.beta1 ** self.t)
        v_db_hat = self.v_db / (1 - self.beta2 ** self.t)


        # Update weights and biases
        self.weights -= learning_rate * m_dw_hat / (np.sqrt(v_dw_hat) + self.epsilon)
        self.bias -= learning_rate * m_db_hat / (np.sqrt(v_db_hat) + self.epsilon)


        self.t += 1

        return dx

class Dropout(Layer):
    def __init__(self,probability=.3):
        self.dropout_probability=probability

    def forward(self,input):
        self.is_dropout= np.random.binomial(1, 1 - self.dropout_probability, size=input.shape[1]) / (1-self.dropout_probability)
        self.is_dropout = np.reshape(-1,1)
        # print(self.is_dropout.shape)
        return input * self.is_dropout.T

    def backward(self, output_gradient, learning_rate):
        return output_gradient*self.is_dropout.T

class Softmax(Layer):
    def forward(self, input):

        # input = (input - np.max(input, axis=1, keepdims=True))
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp, axis=1, keepdims = True)
        return self.output

    def backward(self, output_gradient, learning_rate ):

        z = self.output.T

        # this is cheating knowing that loss is cross_entropy
        return output_gradient * z + z

def cross_entropy(y_true, y_pred):

    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.sum(y_true * np.log(y_pred))
    return loss / len(y_true)

def cross_entropy_prime(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    gradient = -y_true / y_pred
    return gradient.T

def predict(model, input):
    output = input
    for layer in model:
        output = layer.forward(output)

    return output

batch_size=1024

seed = 42
np.random.seed(seed)

indices = np.arange(len(x_train))
np.random.shuffle(indices)

shuffled_data = x_train[indices]
shuffled_labels = y_train[indices]

data_len=len(shuffled_data)
num_batches = (data_len + batch_size - 8) // batch_size

batched_train_set=[]

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, data_len)

    batched_train_set.append((shuffled_data[start_idx:end_idx],shuffled_labels[start_idx:end_idx]))
print(len(batched_train_set))

def batch_train(model, loss, loss_prime, batched_train_set,x_validation, y_validation, epochs=100, verbose=True, learning_rate=0.005, learning_rate_decay = 0.1):

    validation_error = 10
    validation_losses = []
    train_losses = []



    for e in range(epochs):
        error = 0


        lr = (1.0/(1 + e * learning_rate_decay )) * learning_rate

        with tqdm(total=len(batched_train_set), desc=f'Epoch {e + 1}/{epochs}', unit='batch', disable=not verbose) as pbar:
            for x, y in batched_train_set:
                # forward
                output = predict(model, x)

                # error
                error += loss(y, output)

                # backward
                grad = loss_prime(y, output)

                for layer in reversed(model):
                    grad = layer.backward(grad, lr)

                # Update progress bar
                pbar.update(1)

            error /= len(batched_train_set)

            y_validation_pred = predict(model, x_validation)

            new_validation_error = loss(y_validation, y_validation_pred)

            if((new_validation_error-validation_error)/validation_error > .1 ): break

            validation_error = new_validation_error

            validation_error_list.append(validation_error)

            with open(f'./drive/MyDrive/Pikle/dense_model{e}.pkl', 'wb') as file:
                pickle.dump(model, file)

            if verbose:
                pbar.set_postfix(train_error=f'{error:.4f}', val_error=f'{validation_error:.4f}')

    best_e = np.argmin(validation_error_list)
    with open(f'./drive/MyDrive/Pikle/dense_model{best_e}.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
    accuracy(loaded_model, x_test, y_test)

def accuracy(model, x , y):
    y_pred = predict(model, x)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.eye(y.shape[1])[y_pred]

    y_true_flat = y.argmax(axis=1)
    y_pred_flat = y_pred.argmax(axis=1)

    macro_f1 = f1_score(y_true_flat, y_pred_flat, average='macro')

    # Calculate accuracy
    accuracy = accuracy_score(y_true_flat, y_pred_flat)

    print("Macro F1 Score:", macro_f1)
    print("Accuracy:", accuracy)

model = [
    Dense(784, 1024),
    Relu(),
    Dropout(probability=.2),
    Dense(1024, 512),
    Relu(),
    Dropout(probability=.2),
    Dense(512,26),
    Softmax()
]

batch_train(model,cross_entropy,cross_entropy_prime,batched_train_set, x_validation, y_validation, epochs=1000)

accuracy(model, x_test, y_test)

with open('dense_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('dense_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

accuracy(loaded_model, x_test, y_test)

