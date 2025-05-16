#basic setup libraries
import numpy as np
import pandas as pd
from reading_data_helper_functions import read_dataset_file, get_head_with_pandas
from eda_helper_functions import get_dataframe_info, plot_class_count_bar_graph

#plot graph libraries
import seaborn as sns
import matplotlib.pyplot as plt

#standardizing value library
from sklearn.preprocessing import StandardScaler

#resampling tecniques
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

#classification libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

#metrics to measure the classification
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score, classification_report, accuracy_score, roc_curve


def evaluate_classifier(model_values, model, model_name="New Classifier"):
    
    #get variables
    X_train, X_test, y_train, y_test = model_values

    #train the model
    model.fit(X_train, y_train)

    #use to get predictions based on test
    y_pred = model.predict(X_test)

    #evaluate for auc-roc
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except:
        try:
            y_pred_proba = model.decision_function(X_test)
        except AttributeError:
            print("Model does not support probability or decision score. Skipping ROC-AUC.")
            y_pred_proba = None

    #evaluate the classification
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    #print report
    print("")
    print(model_name)
    print(classification_report(y_test, y_pred))
    
    #print the evaluation result
    print(f"Accuracy Score: {accuracy:.3f}")
    print(f"Recall Score: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC AUC Score: {roc_auc:.3f}")

    


def perform_30_70_undersampling(X_train, y_train):

    #make a full data frame for the data used for training
    train_data = pd.concat([X_train, y_train], axis=1)

    #get a random sample from the training data
    number_records_fraud = len(train_data[train_data.Class == 1])
    fraud_indices = np.array(train_data[train_data.Class == 1].index)

    normal_indices = train_data[train_data.Class == 0].index

    new_normal_count = int((70/30) * number_records_fraud)

    random_normal_indices = np.random.choice(normal_indices, new_normal_count, replace = False)
    random_normal_indices = np.array(random_normal_indices)
    
    under_sample_indices = np.concatenate([fraud_indices,random_normal_indices]) 
    under_sample_data = train_data.loc[under_sample_indices,:]
    
    X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
    y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']

    print("normal transactions: ", len(under_sample_data[under_sample_data.Class == 0]))
    print("fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1]))
    print("Total number of transactions in resampled data: ", len(under_sample_data))

    return X_undersample, y_undersample


def perform_70_30_oversampling(X_train, y_train):
    ros = RandomOverSampler(sampling_strategy=0.42, random_state=42)

    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    print("After OverSampling, counts of label '1': {}".format(sum(y_resampled==1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_resampled==0)))

    return X_resampled, y_resampled


def perform_70_30_smote(X_train, y_train):

    sm = SMOTE(sampling_strategy=0.42, random_state=2)
    X_smote, y_smote = sm.fit_resample(X_train, y_train.ravel())

    print("After SMOTE, counts of label '1': {}".format(sum(y_smote==1)))
    print("After SMOTE, counts of label '0': {}".format(sum(y_smote==0)))

    return X_smote, y_smote
