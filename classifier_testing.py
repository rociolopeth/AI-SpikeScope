# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:47:01 2024

@authors: Rocio Lopez Peco, Mikel Val Calvo
@email: yrociro@gmail.es, mikel1982mail@gmail.com
"""

#classiffier

#%%
#load neural data cleaned
from tools.load_data import load_auto_cleaned_data
from tools.waveforms_features import get_durations, check_noise
import numpy as np

mypath = 'C:/Users/yroci/Desktop/analisis_morfologico/all_spontaneous_cleaned/'

units_mean_waveform = load_auto_cleaned_data(mypath)
durations = get_durations(units_mean_waveform)
durations_cleaned, units_mean_waveform_cleaned = check_noise(durations, units_mean_waveform)
durations_cleaned_array = np.array(durations_cleaned).reshape(-1, 1)
#%%
#load model, classify and draw results
from tools.classifier_model import load_model, classifier_model
from tools.visualized_data import draw_classified_waveforms

filename='model.sav'
loaded_model = load_model(filename)

data_frame = classifier_model(loaded_model, durations_cleaned, durations_cleaned_array) 

draw_classified_waveforms(data_frame, units_mean_waveform_cleaned,plot_mean_waveforms=True, plot_violin =True)

