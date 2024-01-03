import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm

import pickle
import copy

import torchvision.datasets as ds
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns

import gc

batch_size=1024

seed = 42
np.random.seed(seed)

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

def predict(model, input,droupout = True):
    output = input
    for layer in model:
        output = layer.forward(output,dropout = droupout)

    return output

def accuracy(model, x , y,droupout=True):
    y_pred = predict(model, x,droupout = droupout)
    y_pred = np.argmax(y_pred, axis=1)

    y_true = y.argmax(axis=1)

    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    print("Macro F1 Score:", macro_f1)
    print("Accuracy:", accuracy)

def get_accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)

    y_true = y_true.argmax(axis=1)

    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy, macro_f1


train_validation_dataset = ds.EMNIST(root='./data', split='letters',
    train=True,
    transform=transforms.ToTensor(),
    download=True)

train_validation_data = []
train_validation_labels = []

# convert to numpy arrays
for data, label in train_validation_dataset:
    data_flattened = data.view(-1)
    train_validation_data.append(data_flattened.numpy())
    train_validation_labels.append(label)

train_validation_data = np.array(train_validation_data)
train_validation_labels = np.array(train_validation_labels)


train_validation_labels=train_validation_labels.reshape(-1,1)

# encode labels
encoder = OneHotEncoder(sparse_output=False)

train_validation_labels_onehot = encoder.fit_transform(train_validation_labels)

with open('./model/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

# split into train and validation sets
validation_size=.15

x_train, x_validation, y_train , y_validation = train_test_split(
    train_validation_data,
    train_validation_labels_onehot,
    test_size=validation_size,
    random_state=42
)

# make train set into batches
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

def batch_train(model, loss, loss_prime, batched_train_set,x_validation, y_validation,model_name='model1', epochs=5, verbose=True, learning_rate=0.005, learning_rate_decay = .5):

    max_validation_f1_score = 0
    validation_losses = []
    train_losses = []
    train_accuracies = []
    validation_accuracies = []
    validation_f1_scores = []

    best_model = None
    
    for e in range(epochs):
        error = 0
        train_accuracy = 0
        
        lr = (1.0/(1 + e * learning_rate_decay )) * learning_rate

        with tqdm(total=len(batched_train_set), desc=f'Epoch {e + 1}/{epochs}', unit='batch', disable=not verbose) as pbar:
            interval_loss = 0
            for x, y in batched_train_set:
                # forward
                output = predict(model, x)

                accuracy, _ = get_accuracy(y, output)
                train_accuracy += accuracy
                
                # error
                error += loss(y, output)
                interval_loss += loss(y, output)

                # backward
                grad = loss_prime(y, output)

                for layer in reversed(model):
                    grad = layer.backward(grad, lr)

                # Update progress bar
                pbar.update(1)
            
            error /= len(batched_train_set)
            train_accuracy /= len(batched_train_set)
            y_validation_pred = predict(model, x_validation,droupout=False)
            validation_loss = loss(y_validation, y_validation_pred)
            
            validation_accuracy, validation_f1_score = get_accuracy(y_validation, y_validation_pred)
            if validation_f1_score > max_validation_f1_score:
                if best_model is not None:
                    del best_model
                max_validation_f1_score = validation_f1_score
                best_model = copy.deepcopy(model)
            
            train_losses.append(error)
            validation_losses.append(validation_loss)
            train_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)
            validation_f1_scores.append(validation_f1_score)

            
            if verbose:
                pbar.set_postfix(train_error=f'{error:.4f}', val_error=f'{validation_loss:.4f}')
            if (e+1) % 10 == 0:
                gc.collect()
    # with open(f'./model/{model_name}.pickle', 'wb') as f:
    #     pickle.dump(best_model, f)
    
    # dump only weights
    weights = []
    for layer in best_model:
        if isinstance(layer, Dense):
            weights.append((layer.weights, layer.bias))
    with open(f'./model/{model_name}.pickle', 'wb') as f:
        pickle.dump(weights, f)
    
    y_validation_pred = predict(best_model, x_validation,droupout=False)
    y_validation_pred = np.argmax(y_validation_pred, axis=1)
    
    y_validation_true = np.argmax(y_validation, axis=1)
    
    cm = confusion_matrix(y_validation_true, y_validation_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name}-{learning_rate} - Confusion Matrix')
    plt.savefig(f'./graph/{model_name}-confusion_matrix.png')
    plt.close()
    
    best_e = np.argmax(validation_f1_scores)
    print(f'''Best epoch: {best_e + 1}, validation accuracy: {validation_accuracies[best_e]:.4f}, validation f1 score: {validation_f1_scores[best_e]:.4f}, model name: {model_name},
          train loss: {train_losses[best_e]:.4f}, validation loss: {validation_losses[best_e]:.4f}, train accuracy: {train_accuracies[best_e]:.4f}, validation accuracy: {validation_accuracies[best_e]:.4f}''')
    
    return train_losses, validation_losses, train_accuracies, validation_accuracies, validation_f1_scores



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

batch_train(model,cross_entropy,cross_entropy_prime,batched_train_set, x_validation, y_validation, epochs=100,
            learning_rate=.005, model_name='ChoosenBestModel')

# model_names = ['my_model3']

# models =[
#     #    [ Dense(784, 512),
#     #     Relu(),
#     #     Dropout(probability=.15),
#     #     Dense(512, 256),
#     #     Relu(),
#     #     Dropout(probability=.15),
#     #     Dense(256, 128),
#     #     Relu(),
#     #     Dropout(probability=.10),
#     #     Dense(128, 64),
#     #     Relu(),
#     #     Dropout(probability=.10),
#     #     Dense(64, 26),
#     #     Softmax()
#     # ],
#     #    [
#     #     Dense(784, 2048),
#     #     Relu(),
#     #     Dropout(probability=.4),
#     #     Dense(2048, 1024),
#     #     Relu(),
#     #     Dropout(probability=.3),
#     #     Dense(1024, 26),
#     #     Softmax()
#     # ], 
#     [
#         Dense(784, 1024),
#         Relu(),
#         Dropout(probability=.2),
#         Dense(1024, 512),
#         Relu(),
#         Dropout(probability=.2),
#         Dense(512,26),
#         Softmax()
#     ]
# ]

# learning_rates = [.01, .005, .001,.0005]
# # learning_rates = [.01]

# for model, model_name in zip(models, model_names):
    
#     train_output = predict(model, x_train, droupout=False)
#     random_loss = cross_entropy(y_train, train_output)
#     train_accuracy, _ = get_accuracy(y_train, train_output)
    
#     validation_output = predict(model, x_validation, droupout=False)
#     validation_loss = cross_entropy(y_validation, validation_output)
#     validation_accuracy, validation_f1 = get_accuracy(y_validation, validation_output)
    
#     for learning_rate in learning_rates:
#         network = copy.deepcopy(model)
#         train_losses, validation_losses, train_accuracies, validation_accuracies, validation_f1_scores= batch_train(
#             network, 
#             cross_entropy,
#             cross_entropy_prime,
#             batched_train_set,
#             x_validation,
#             y_validation, 
#             model_name=f'{model_name}_{learning_rate}', epochs = 50, learning_rate=learning_rate)
        
    
#         train_losses = [random_loss]+train_losses
#         validation_losses = [validation_loss]+validation_losses
#         train_accuracies = [train_accuracy]+train_accuracies
#         validation_accuracies = [validation_accuracy]+validation_accuracies
#         validation_f1_scores = [validation_f1]+validation_f1_scores
    
        
#         fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#         axes = axes.flatten()

#         for i, performance_measure in enumerate(['loss', 'accuracy', 'f1']):
#             ax = axes[i]
#             ax.set_title(f'{model_name} - {performance_measure.capitalize()}-Learning Rate={learning_rate}')

#             if performance_measure == 'loss':
#                 ax.plot(train_losses, label=f'Train LR={learning_rate}')
#                 ax.plot(validation_losses, label=f'Validation LR={learning_rate}')
#                 ax.set_ylabel('Loss')
#             elif performance_measure == 'accuracy':
#                 ax.plot(train_accuracies, label=f'Train LR={learning_rate}')
#                 ax.plot(validation_accuracies, label=f'Validation LR={learning_rate}')
#                 ax.set_ylabel('Accuracy')
#             elif performance_measure == 'f1':
#                 ax.plot(validation_f1_scores, label=f'Validation F1 LR={learning_rate}')
#                 ax.set_ylabel('F1 Score')

#             ax.set_xlabel('Epochs')
#             ax.legend()
#         # Save the figure
#         fig.savefig(f'./graph/{model_name}_{learning_rate}.png')
        
        
