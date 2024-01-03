import torchvision.datasets as ds
import torchvision.transforms as transforms

import numpy as np

import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix



class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.is_dropout=None
        self.dropout_probability=.3
        self.weights = None
        self.bias = None

    def forward(self, input, dropout = True):
        # forward output from the input
        pass

    def backward(self, output_gradient, learning_rate):
        # update its parameter and return values for the previous level
        pass

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input, dropout = True):
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


    def forward(self, input, dropout = True):
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

    def forward(self,input, dropout = True):
        if dropout:
            self.is_dropout= np.random.binomial(1, 1 - self.dropout_probability, size=input.shape[1]) / (1-self.dropout_probability)
            self.is_dropout = self.is_dropout.reshape(-1,1)
            return input*self.is_dropout.T
        else:
            return input

    def backward(self, output_gradient, learning_rate):
        return output_gradient*self.is_dropout

class Softmax(Layer):
    def forward(self, input, dropout = True):

        # input = (input - np.max(input, axis=1, keepdims=True))
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp, axis=1, keepdims = True)
        return self.output

    def backward(self, output_gradient, learning_rate ):

        z = self.output.T

        # this is cheating knowing that loss is cross_entropy
        return output_gradient * z + z

def load_raw_data(data_path, level_path):
    dataset = ds.EMNIST(root=data_path,
        split='letters',
        train=False,
        )
    x = []
    labels = []
    for data, label in dataset:
        data = np.array(data)
        x.append(data.flatten())
        labels.append(label)
    labels = np.array(labels).reshape(-1,1)
    x = np.array(x) / 255
    
    return x, labels

def test_emnist_independent_testset():
    independent_test_dataset = ds.EMNIST(root='./data',
        split='letters',
        train=False,
        transform=transforms.ToTensor())
    independent_test_data = []
    independent_test_labels = []

    for data, label in independent_test_dataset:
        data_flattened = data.view(-1)
        independent_test_data.append(data_flattened.numpy())
        independent_test_labels.append(label)

    independent_test_data = np.array(independent_test_data)
    independent_test_labels = np.array(independent_test_labels)
    
    independent_test_labels = independent_test_labels.reshape(-1,1)
    return independent_test_data, independent_test_labels

def encoding(independent_test_data, independent_test_labels):
    # with open('./model/encoder.pickle', 'rb') as f:
    #     encoder = pickle.load(f)

    # independent_test_labels = encoder.transform(independent_test_labels)
    
    independent_test_labels = independent_test_labels - 1
    independent_test_labels = np.eye(26)[independent_test_labels.reshape(-1)]
    
    x_test=independent_test_data
    y_test=independent_test_labels
    return x_test, y_test

def load_model(best_model_path):
    with open(best_model_path, 'rb') as f:
        best_model = pickle.load(f)
    return best_model

def predict(model, input,droupout = True):
    output = input
    for layer in model:
        output = layer.forward(output,dropout = droupout)

    return output
def load_weights(model, pickle_file):
    with open(pickle_file, 'rb') as f:
        weights = pickle.load(f)
    i = 0
    for layer in model:
        if isinstance(layer, Dense):
            layer.weights , layer.bias = weights[i]
            i += 1
    return model

def cross_entropy(y_true, y_pred):

    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.sum(y_true * np.log(y_pred))
    return loss / len(y_true)

def test_model(best_model, x_test, y_test):
    output = predict(best_model, x_test, droupout = False)
    
    test_loss = cross_entropy(y_test, output)
    print("Test Loss: ", test_loss)
    
    y_pred = np.argmax(output, axis=1)
    
    y_true = y_test.argmax(axis=1)
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    print("Test Accuracy: ", accuracy)
    print("Test Macro F1 Score: ", macro_f1)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix of independent test set')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    # save confusion matrix
    plt.savefig('independent_test_set_confusion_matrix.png')
    
model = [
    Dense(784, 1024),
    Relu(),
    Dropout(probability=.3),
    Dense(1024, 512),
    Relu(),
    Dropout(probability=.3),
    Dense(512,26),
    Softmax()
 ]
data, labels = load_raw_data('./data', './data')
x_test, y_test = encoding(data, labels)
best_model = load_weights(model, 'model_1805091.pickle')
test_model(best_model, x_test, y_test)