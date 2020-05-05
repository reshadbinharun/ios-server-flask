import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

#Example Use
# x_path = 'training_data/pH/sensitivity_csv_files/x_train/'
# y_path = 'training_data/pH/sensitivity_csv_files/y_train/'
# x_path_list = sorted(os.listdir(x_path))
# sense_y_train, sense_x_train, sense_y_test, sense_x_test = create_data_vectors(x_path_list, y_path_list, x_path, y_path)

# Creates training and test split from image set
def create_data_vectors(x_path_list, y_path_list, x_path, y_path):
    X_list = []
    y_list = []
    for xfile, yfile in zip(x_path_list, y_path_list):
        X_list.append(np.asarray(pd.read_csv(x_path + xfile)))
        y_list.append(np.asarray(pd.read_csv(y_path + yfile)))
    X_aug = np.vstack(np.asarray(X_list))
    Y_aug = np.vstack(np.asarray(y_list))
    X_Z = StandardScaler().fit_transform(X_aug)
    data = np.concatenate((Y_aug, X_Z), axis = 1)
    train, test = train_test_split(data, test_size=0.1)

    train_outcome = train[:,0]
    train_predictors = train[:,1:5851]
    test_outcome = test[:,0]
    test_predictors = test[:,1:5851]
    return train_outcome, train_predictors, test_outcome, test_predictors

# Evaluate MLP performance on Training Set
def train_eval_mlp(k_folds=5):
	train_MAEs = [] 
	val_MAEs = []

	new_train_predictors = np.append(train_predictors, sense_x_train, axis=0)
	new_train_outcomes = np.append(train_outcome, sense_y_train, axis=0)
	new_test_predictors = np.append(test_predictors, sense_x_test, axis=0)
	new_test_outcomes = np.append(test_outcome, sense_y_test, axis=0)

	kf = KFold(n_splits=k_folds)
	for train_index, val_index in kf.split(new_train_predictors):
	    x_train, x_val = new_train_predictors[train_index], new_train_predictors[val_index]
	    y_train, y_val = new_train_outcomes[train_index], new_train_outcomes[val_index]
	    pca = PCA(n_components=250)
	    pca.fit(x_train)
	    
	    aggregate_train_PCs = pca.transform(x_train)
	    aggregate_val_PCs = pca.transform(x_val)
	    model = MLPRegressor(solver='lbfgs', alpha=10, activation='tanh', max_iter=8000)
	    model.fit(aggregate_train_PCs, y_train)
	    train_pred = model.predict(aggregate_train_PCs)
	    train_MAEs.append(mean_absolute_error(y_train, train_pred))
	    val_pred = model.predict(aggregate_val_PCs)
	    val_MAEs.append(mean_absolute_error(y_val, val_pred))
	return train_MAEs, val_MAEs

# Evaluate SVM performance on Training Set
def train_eval_svm(k_folds=5):
	train_MAEs = [] 
	val_MAEs = []

	new_train_predictors = np.append(train_predictors, sense_x_train, axis=0)
	new_train_outcomes = np.append(train_outcome, sense_y_train, axis=0)
	new_test_predictors = np.append(test_predictors, sense_x_test, axis=0)
	new_test_outcomes = np.append(test_outcome, sense_y_test, axis=0)

	kf = KFold(n_splits=k_folds)
	for train_index, val_index in kf.split(new_train_predictors):
	    x_train, x_val = new_train_predictors[train_index], new_train_predictors[val_index]
	    y_train, y_val = new_train_outcomes[train_index], new_train_outcomes[val_index]
	    pca = PCA(n_components=250)
	    pca.fit(x_train)
	    
	    aggregate_train_PCs = pca.transform(x_train)
	    aggregate_val_PCs = pca.transform(x_val)
	    model = SVR(C=10, epsilon=0.001, tol=0.001)
	    model.fit(aggregate_train_PCs, y_train)
	    train_pred = model.predict(aggregate_train_PCs)
	    train_MAEs.append(mean_absolute_error(y_train, train_pred))
	    val_pred = model.predict(aggregate_val_PCs)
	    val_MAEs.append(mean_absolute_error(y_val, val_pred))
	return train_MAEs, val_MAEs, 

# Function to predict pH value of taken image
#def predict(img = []):























