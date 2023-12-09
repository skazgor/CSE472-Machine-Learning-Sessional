import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import  confusion_matrix

# Make NumPy printouts easier to read.
np.set_printoptions(precision=5, suppress=True)

def preprocess_telco_customer_churn(num_features=20, seed = 42, view_step=False):

    df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    if view_step:
        print(df.info())
        print(df.sample(1))
    
    df.drop(columns='customerID', inplace= True)
    df = df[pd.to_numeric(df['TotalCharges'], errors='coerce').notna()]
    df['TotalCharges']= df['TotalCharges'].astype(float) 

    # df.duplicated().sum()
    # df.drop_duplicates(inplace=True)

    categorical_columns = df.select_dtypes(include=['object']).columns
    df_encoded = df.copy()

    for col in categorical_columns:
        if df[col].nunique() == 2:
            possible_labels = df[col].unique()

            label_dict = {}
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            df_encoded[col] = df[col].replace(label_dict)
        else:
            df_temp = pd.get_dummies(df[col], prefix=col)
            df_encoded = pd.concat([df_encoded, df_temp], axis=1)
            df_encoded.drop(columns=[col], inplace=True)

    if view_step:
        print(df_encoded.sample(1))
    
    df=df_encoded

    X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                  df.Churn.values,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify=df.Churn.values)
    
    df['data_type'] = ['not_set']*df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'test'
    train_data = df[df.data_type == 'train']

    # Normalize the data
    columns=['tenure','MonthlyCharges','TotalCharges']
    for col in columns:
        mean_value = train_data[col].mean()
        std_value = train_data[col].std()

        df[col] = (df[col] - mean_value) / std_value
    if view_step:
        print(df.sample(1))

    train_df = df[df['data_type'] == 'train'].copy()
    test_df = df[df['data_type'] == 'test'].copy()
    # Drop 'data_type' from both sets
    train_df.drop(columns=['data_type'], inplace=True)
    test_df.drop(columns=['data_type'], inplace=True)
    x_train=train_df.drop('Churn',axis=1)
    y_train=train_df['Churn']
    x_test=test_df.drop('Churn',axis=1)
    y_test=test_df['Churn']

    N=num_features
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    selector.fit(x_train, y_train)
    feature_scores = pd.DataFrame({'Feature': x_train.columns, 'Information_Gain': selector.scores_})
    feature_scores = feature_scores.sort_values(by='Information_Gain', ascending=False)
    selected_features = feature_scores.head(N)['Feature'].tolist()
    x_train_subset = x_train[selected_features]
    x_test_subset = x_test[selected_features]
    if view_step:
        print(x_train_subset.sample(1))

    train_x=x_train_subset.values
    train_y = y_train.values

    test_x = x_test_subset.values
    test_y = y_test.values

    train_y=train_y.reshape(-1, 1)
    test_y=test_y.reshape(-1,1)
    return train_x,train_y,test_x,test_y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_weights(num_features):
    return np.zeros((num_features, 1))

def add_intercept_column(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)

def accuracy_with_intercept(X,Y, weights):
    predictions = sigmoid(np.dot(X, weights))
    y_pred=(predictions >= 0.5).astype(int)
    accuracy_ = np.mean(predictions == Y)
    return accuracy_

def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000, error_rate=.5):
    X = add_intercept_column(X)
    m, n = X.shape
    weights = initialize_weights(n)

    for i in range(num_iterations):
        if 1-accuracy_with_intercept(X,y,weights)<error_rate:
            return weights
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        error = predictions - y
        gradient = np.dot(X.T, error) / m
        weights -= learning_rate * gradient

    return weights

def AdaBoost(X,y,l_weak,K,seed=42,error_rate=.5):
    np.random.seed(seed)

    intercepted_X=add_intercept_column(X)
    N,temp=X.shape

    w = np.ones(N) / N
    w = w.reshape(-1, 1)

    h=[]
    weights=np.zeros(K)

    for k in range(K):
        samples_index=np.random.choice(N, size=N, replace=True, p=w.flatten())

        samples_x=X[samples_index]
        samples_y= y[samples_index]

        h_k = l_weak(samples_x,samples_y)

        z = np.dot(intercepted_X, h_k)
        predictions_probability=sigmoid(z)
        predictions_lebel=(predictions_probability >= 0.5).astype(int)

        error=np.sum(np.dot(w.T,predictions_lebel!=y))

        print(error)

        if error> .5:
            continue

        for i in range(N):
            if predictions_lebel[i] == y[i]:
                w[i]=w[i]*error/(1-error)
        w=w/np.sum(w)

        h.append(h_k)
        weights[k]=np.log((1 - error) / max(error, 1e-10))
    return h,weights

def weighted_majority(h, z, x):
    votes=np.zeros((x.shape[0],1))

    x=add_intercept_column(x)
    for h_z, z_k in zip(h,z):

        z_values=np.dot(x,h_z)
        predictions_probability=sigmoid(z_values)
        predictions_lebel=(predictions_probability >= 0.5).astype(int)

        predictions_lebel[predictions_lebel == 0] = -1
        votes+=predictions_lebel*z_k
    return (votes>0 ).astype(int)

def performacne_analysis(y_true,y_pred):
    conf_matrix = confusion_matrix(y_true.flatten(), y_pred.flatten())
    TN, FP, FN, TP = conf_matrix.ravel()

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # True Positive Rate (Recall)
    recall = TP / (TP + FN)

    # True Negative Rate (Specificity)
    specificity = TN / (TN + FP)

    # Positive Predictive Value (Precision)
    precision = TP / (TP + FP)

    # False Discovery Rate
    fdr = FP / (TP + FP)

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall (True Positive Rate): {recall:.4f}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"Precision (Positive Predictive Value): {precision:.4f}")
    print(f"False Discovery Rate: {fdr:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

def run_telco_customer_churn(num_features=20, seed = 42, view_step=False, K=10):
    train_x,train_y,test_x,test_y=preprocess_telco_customer_churn(num_features=num_features, seed = seed, view_step=view_step)

    h,z=AdaBoost(train_x,train_y,logistic_regression,K, seed=seed)
    y_pred=weighted_majority(h,z,test_x)

    performacne_analysis(test_y,y_pred)

run_telco_customer_churn(num_features=20, seed = 42, view_step=True, K=10)
