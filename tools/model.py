# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:46:55 2024

@authors: Rocio Lopez Peco, Mikel Val Calvo
@email: yrociro@gmail.es, mikel1982mail@gmail.com
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
 
def train_test_model(train_durations,labels, filename, plot_boxplot = True):
    
    # dataframe to train the model
    data = {'Durations': train_durations, 'Labels': labels}
    df = pd.DataFrame(data)
    
    #Adding test data to later evaluate the trainig
    X_train, X_test, y_train, y_test = train_test_split(df["Durations"], df["Labels"], test_size=0.33, random_state=42)
    X_train = np.array(X_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    
    '''RandomForestClassifier'''
    
    forest_model = RandomForestClassifier(random_state=1)
    forest_model.fit(X_train, y_train)
    machine_preds = forest_model.predict(X_test)
    
    if plot_boxplot:
        df.boxplot(column='Durations', by= 'Labels') 
    
    # Evalute
    print(classification_report(y_test, machine_preds))
    
    pickle.dump(forest_model, open(filename, 'wb'))
    
    
def classifier_model(forest_model, durations_cleaned, durations_cleaned_array, plot_boxplot = True):
    machine_preds_dataframe = forest_model.predict(durations_cleaned_array)
    data_frame_dict = {'Durations': durations_cleaned, 'Labels': machine_preds_dataframe}
    data_frame = pd.DataFrame(data_frame_dict)
    if plot_boxplot:
        data_frame.boxplot(column='Durations', by= 'Labels') 
    
    return data_frame
    
def load_model(filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

    