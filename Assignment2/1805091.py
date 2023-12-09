# !mkdir -p ~/.kaggle
# !mv kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d mlg-ulb/creditcardfraud
# !unzip creditcardfraud.zip
# !kaggle datasets download -d blastchar/telco-customer-churn
# !unzip telco-customer-churn.zip
# !curl -O https://archive.ics.uci.edu/static/public/2/adult.zip
# !unzip adult.zip



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import  confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# import matplotlib.pyplot as plt
# import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=5, suppress=True)

# def plot_count(df):
#     categorical_columns = df.select_dtypes(include=['object']).columns

#     fig, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(10, 4 * len(categorical_columns)))

#     for i, column in enumerate(categorical_columns):
#         sns.countplot(x=column, data=df, ax=axes[i])
#         axes[i].set_title(f"Count of {column}")

#     plt.tight_layout()
#     plt.show()

def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def information_gain(X, y, feature_index):
    total_entropy = entropy(y)

    values = np.unique(X[:, feature_index])

    weighted_entropy = 0
    for value in values:
        subset_indices = X[:, feature_index] == value
        subset_entropy = entropy(y[subset_indices])
        weighted_entropy += len(y[subset_indices]) / len(y) * subset_entropy

        return total_entropy - weighted_entropy

def select_best_features(X, y, num_features):
    num_samples, num_features_total = X.shape
    information_gains = []

    X_np=X.values
    Y_np= y.values

    for feature_index,col in zip(range(num_features_total),X.columns):
        gain = information_gain(X_np, Y_np, feature_index)
        information_gains.append((col, gain))

    information_gains.sort(key=lambda x: x[1], reverse=True)

    selected_features = [feature_index for feature_index, _ in information_gains[:num_features]]
    return selected_features
def level_encode_for_binary_class(df):
    categorical_columns = df.select_dtypes(include=['object']).columns

  # Create a new DataFrame to store encoded columns
    df_encoded = df.copy()

    for col in categorical_columns:
        if df[col].nunique() == 2:
            possible_labels = df[col].unique()

            label_dict = {}
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
      
            df_encoded[col] = df[col].replace(label_dict)
    return df_encoded

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
                                                  random_state=seed,
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
    
    # selector = SelectKBest(score_func=mutual_info_classif, k='all')
    # selector.fit(x_train, y_train)
    # feature_scores = pd.DataFrame({'Feature': x_train.columns, 'Information_Gain': selector.scores_})
    # feature_scores = feature_scores.sort_values(by='Information_Gain', ascending=False)
    # selected_features = feature_scores.head(N)['Feature'].tolist()
    
    
    selected_features=select_best_features(x_train, y_train,N)
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

def preprocess_adult(num_features=20, seed = 42, view_step=False):
    header_names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income'
    ]
    train_df=pd.read_csv("adult.data",header=None,names=header_names)
    test_df=pd.read_csv("adult.test",header=None,names=header_names,skiprows=1)
    
    if view_step:
        print(train_df.info())
        print(train_df.sample(1))
    
    columns=train_df.select_dtypes(exclude=['object']).columns
    # normalization
    for col in columns:
        mean_value = train_df[col].mean()
        std_value = train_df[col].std()

        train_df[col] = (train_df[col] - mean_value) / std_value
        test_df[col] = (test_df[col]-mean_value)/std_value
    
    if view_step:
        print(train_df.sample(1))
    
    # plot_count(train_df)
    
    categorical_columns_with_missing = ['workclass', 'occupation', 'native-country']
    for column in categorical_columns_with_missing:
        non_missing_values = train_df[train_df[column] != ' ?'][column]
        probabilities = non_missing_values.value_counts(normalize=True)

        missing_values_count_train = (train_df[column] == ' ?').sum()
        missing_values_count_test = (test_df[column] == ' ?').sum()

        random_sample_train = np.random.choice(probabilities.index, size=missing_values_count_train, p=probabilities.values)
        random_sample_test = np.random.choice(probabilities.index, size=missing_values_count_test, p=probabilities.values)

        train_df.loc[train_df[column] == ' ?', column] = random_sample_train
        test_df.loc[test_df[column] == ' ?', column] = random_sample_test
    
    # plot_count(train_df)
    train_df.drop(columns=['education'], inplace=True)
    test_df.drop(columns=['education'], inplace=True)
    
    if view_step:
        print(train_df.sample(1))
    
    train_df=level_encode_for_binary_class(train_df)
    test_df=level_encode_for_binary_class(test_df)
    
    
    # One hot encoding
    others=.05
    df=train_df.copy()
    df_test=test_df.copy()
    categorical_columns = df.select_dtypes(include=['object']).columns
    threshold_count = len(df) * (1-others)
    
    for column in categorical_columns:
        top_categories = df[column].value_counts()
        num_categories_to_include = (top_categories.cumsum() <= threshold_count).sum()+1

        encoder = OneHotEncoder(max_categories=num_categories_to_include+1 ,sparse_output=False)
        # encoder = OneHotEncoder(sparse_output=False)

        encoded_categories = encoder.fit_transform(df[[column]])
        encoded_categories_test = encoder.fit_transform(df_test[[column]])

        encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out([column]))
        encoded_df_test=pd.DataFrame(encoded_categories_test, columns=encoder.get_feature_names_out([column]))

        df=pd.concat([df,encoded_df],axis=1)
        df.drop(columns=[column], inplace=True)

        df_test=pd.concat([df_test,encoded_df_test],axis=1)
        df_test.drop(columns=[column],inplace=True)

    
    if view_step:
        print(df.sample(1))
    
    x_train=df.drop('income',axis=1)
    y_train=df['income']

    x_test=df_test.drop('income',axis=1)
    y_test=df_test['income']
    
    N=num_features
    selected_features=select_best_features(x_train, y_train,N)
    
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

def preprocess_credit_card_fraud(num_features=20, seed = 42, view_step=False):
    df=pd.read_csv('creditcard.csv')
    
    if view_step:
        print(df.info())
        print(df.sample(1))
        df.isna().sum()
    
    true_df=df[df.Class==1]

    # Probability of selecting 0 data
    probability=(df.Class == 0).astype(int)
    probability=probability/probability.sum()
    
    np.random.seed(42)

    samples_index = np.random.choice(df.index, size=20000, replace=False, p=probability)

    df_sampled = df.loc[samples_index]

    df_combined = pd.concat([df_sampled, true_df])
    df= df_combined
    
    if view_step:
        print(df.sample(1))
    
    X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                  df.Class.values,
                                                  test_size=0.2,
                                                  random_state=seed,
                                                  stratify=df.Class.values)
    df.drop(columns=['Class'],inplace=True)
    
    train_df=df.loc[X_train]
    test_df = df.loc[X_val]
    
    # normalization
    for col in df.columns:
        mean_value = train_df[col].mean()
        std_value = train_df[col].std()

        train_df[col] = (train_df[col] - mean_value) / std_value
        test_df[col] = (test_df[col]-mean_value)/std_value
        
    if view_step:
        print(train_df.sample(1))
    
    x_train=train_df
    y_train

    x_test=test_df
    y_test=y_val
    
    N=num_features
    selected_features=select_best_features(x_train,pd.DataFrame({'Class':y_train}),N)
    
    x_train_subset = x_train[selected_features]
    x_test_subset = x_test[selected_features]
    
    if view_step:
        print(x_train_subset.sample(1))
    train_x=x_train_subset.values
    train_y = y_train

    test_x = x_test_subset.values
    test_y = y_test

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
    accuracy_ = np.mean(y_pred == Y)
    return accuracy_

def logistic_regression(X, y, learning_rate=0.1, num_iterations=10000, error_rate=.5):
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

def AdaBoost(X,y,l_weak,K,seed=42,error_rate=.5, print_error=True):
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

        h_k = l_weak(samples_x,samples_y,error_rate=error_rate)

        z = np.dot(intercepted_X, h_k)
        predictions_probability=sigmoid(z)
        predictions_lebel=(predictions_probability >= 0.5).astype(int)

        error=np.sum(np.dot(w.T,predictions_lebel!=y))

        if print_error:
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

    # print(f"Confusion Matrix:\n{conf_matrix}")
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    recall = TP / (TP + FN)

    specificity = TN / (TN + FP)

    precision = TP / (TP + FP)

    fdr = FP / (TP + FP)

    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall (True Positive Rate): {recall:.4f}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"Precision (Positive Predictive Value): {precision:.4f}")
    print(f"False Discovery Rate: {fdr:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

def run_adaboost(dataset,train_x,train_y,test_x,test_y, num_features=20, seed = 42, view_step=False, K=10,error_rate=.5):
    print("Dataset:"+dataset+ "\n" + "Number of features: "+ str(num_features) +"\n" + "K: "+ str(K) )
    
    h,z=AdaBoost(train_x,train_y,logistic_regression,K, seed=seed,print_error=view_step,error_rate=error_rate)
    
    print("Performance on training set:")
    y_pred=weighted_majority(h,z,train_x)
    performacne_analysis(train_y,y_pred)

    print("Performance on test set:")
    y_pred=weighted_majority(h,z,test_x)
    performacne_analysis(test_y,y_pred)
    print("")

def run_telco_customer_churn(num_features=20, seed = 42, view_step=False, K=10, error_rate=.5):
    train_x,train_y,test_x,test_y=preprocess_telco_customer_churn(num_features=num_features, seed = seed, view_step=view_step)

    run_adaboost("Telco Customer Churn",train_x,train_y,test_x,test_y, num_features=num_features, seed = seed, view_step=view_step, K=K, error_rate=error_rate)

def run_adult(num_features=20, seed = 42, view_step=False, K=10, error_rate=.5):
    train_x,train_y,test_x,test_y=preprocess_adult(num_features=num_features, seed = seed, view_step=view_step)

    run_adaboost("Adult",train_x,train_y,test_x,test_y, num_features=num_features, seed = seed, view_step=view_step, K=K, error_rate=error_rate)

def run_credit_card_fraud(num_features=20, seed = 42, view_step=False, K=10, error_rate=.5):
    train_x,train_y,test_x,test_y=preprocess_credit_card_fraud(num_features=num_features, seed = seed, view_step=view_step)

    run_adaboost("Credit Card Fraud",train_x,train_y,test_x,test_y, num_features=num_features, seed = seed, view_step=view_step, K=K, error_rate=error_rate)


def run_all():
    k=[5,10,15,20]
    # train_x_telco,train_y_telco,test_x_telco,test_y_telco=preprocess_telco_customer_churn(num_features=20, seed = 42, view_step=False)
    # train_x_adult,train_y_adult,test_x_adult,test_y_adult=preprocess_adult(num_features=20, seed = 42, view_step=False)
    # train_x_credit,train_y_credit,test_x_credit,test_y_credit=preprocess_credit_card_fraud(num_features=20, seed = 42, view_step=False)
    for i in k:
        run_telco_customer_churn( num_features=20, seed = 42, view_step=False, K=i,error_rate=.5)
        run_adult( num_features=20, seed = 42, view_step=False, K=i,error_rate=.5)
        run_credit_card_fraud( num_features=15, seed = 42, view_step=False, K=i,error_rate=.5)


# run_telco_customer_churn(num_features=20, seed = 42, view_step=True, K=10)
# run_adult(num_features=20, seed = 42, view_step=True, K=10)
# run_credit_card_fraud(num_features=20, seed = 42, view_step=True, K=10)

stutus=input("Do you want to choose spcific dataset? or run all? (y/n): ")
if stutus=='y':
    dataset=input("Enter dataset name(telco, adult, credit): ")
    num_features=int(input("Enter number of features: "))
    K=int(input("Enter K: "))
    # view_step=input("Do you want to view steps? (y/n): ")
    # if view_step=='y':
    #     view_step=True
    # else:
    #     view_step=False
    error_rate=float(input("Enter error rate: "))
    if dataset=='telco':
        run_telco_customer_churn(num_features=num_features, seed = 42, view_step=True, K=K, error_rate=error_rate)
    elif dataset=='adult':
        run_adult(num_features=num_features, seed = 42, view_step=True, K=K, error_rate=error_rate)
    elif dataset=='credit':
        run_credit_card_fraud(num_features=num_features, seed = 42, view_step=True, K=K, error_rate=error_rate)
    else:
        print("Invalid dataset name")
else :
    run_all()

