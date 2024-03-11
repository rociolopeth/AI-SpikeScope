# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:11:00 2024

@authors: Rocio Lopez Peco, Mikel Val Calvo
@email: yrociro@gmail.es, mikel1982mail@gmail.com
"""
import numpy as np


def get_durations(new_units_all):
    x_max_all = []  
    
    for array in new_units_all:
        for value in array:
            x_max = np.argmax(array)
        x_max_all.append(x_max)
    
    # duration = (abs(x_max-x_min) / ((length-1)*10/length))/30 # estamos pasando muestras del vector interpolado a milisegundos
    
    # x_resta = []
    value_rest = []
    durations = []
    for i in x_max_all:
        value_rest = i -12
        duration = value_rest*1000/30000
        durations.append(duration)
    return durations

def check_noise (durations, units_mean_waveform):
    durations_cleaned = []
    units_mean_waveform_cleaned = []
    for i in range (len(durations)):
        if durations[i] > 0 and durations[i] < 0.6:
            durations_cleaned.append(durations[i])
            units_mean_waveform_cleaned.append(units_mean_waveform[i])
    return durations_cleaned, units_mean_waveform_cleaned
