# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:10:45 2024

@authors: Rocio Lopez Peco, Mikel Val Calvo
@email: yrociro@gmail.es, mikel1982mail@gmail.com
"""

# classifier training

#%% load spike templates
from tools.load_data import extract_data

mypath =  'C:/Users/yroci/Desktop/analisis_morfologico/templates/'

new_units_all = extract_data(mypath)

#%% cluster data
from tools.clustering_data import UMAP_reducer, Silhouette_visualizer, GMM_classifier

embedding = UMAP_reducer(new_units_all, plot_UMAP = True)

Silhouette_visualizer(embedding)

labels = GMM_classifier(embedding, plot_clustering =True)

#%%  visualize clustered data
from tools.visualized_data import draw_clustered_waveforms, features_clustered_waveforms

waveforms_params = draw_clustered_waveforms(new_units_all, labels)

features_clustered_waveforms(waveforms_params, plot_features = True)

#%% clasifier training
from tools.waveforms_features import get_durations
from tools.classifier_model import train_test_model

train_durations = get_durations(new_units_all)

filename='model.sav'
train_test_model(train_durations,labels,filename) 
