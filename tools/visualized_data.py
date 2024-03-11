# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:13:23 2024

@authors: Rocio Lopez Peco, Mikel Val Calvo
@email: yrociro@gmail.es, mikel1982mail@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter



def draw_clustered_waveforms(new_units_all, labels):
    new_units_all = np.array(new_units_all)
    waveforms_params = []
    colors = ['purple','green', 'blue']
    groups = ['Group 1', 'Group 3', 'Group 2']

    plt.figure()
    plt.title('Units waveforms', fontsize =22)
    plt.xlabel('Time(ms)', fontsize=18)
    plt.grid(b=None)
    for it,ID in enumerate(np.unique(labels)):
        waveforms = new_units_all[labels == ID]
        plt.plot(waveforms.mean(axis=0), c=colors[it], label=groups[it])
        plt.xticks(np.arange(0,49,5), np.round(np.arange(0,1.6,.165),2))
        plt.fill_between( np.arange(0,48),waveforms.mean(axis=0) - waveforms.std(axis=0), waveforms.mean(axis=0) + waveforms.std(axis=0), alpha=.4, color=colors[it])
        waveforms_params.append(waveforms.mean(axis=0))
        plt.legend()
    return waveforms_params

def features_clustered_waveforms(waveforms_params, plot_features = True):
    for j in range(len(waveforms_params)):    
        unit = waveforms_params[j]
        unit = unit.tolist()
        
        baseline = np.mean(unit[:3])
        unit = unit - baseline
           
        # le hacemos el valor absoluto
        abs_unit = abs(unit)
        abs_unit = abs_unit / max(abs_unit)
        abs_unit = savgol_filter(abs_unit, 11, 2)
        
        # calculamos los dos picos
        peaks, _ = find_peaks(abs_unit, height=.3)
        
        amplitude = abs(baseline - unit[peaks[1]]) + abs(baseline - unit[peaks[0]])
        round(amplitude, 2)
        ratio = abs(baseline - unit[peaks[1]]) / abs(baseline - unit[peaks[0]])
        round(ratio, 2)
        duration = abs(peaks[0]-peaks[1])/30
        round(duration, 2)
        
        if plot_features:
            plt.figure()
            plt.plot(unit)
            plt.title(f"{j} group")
            plt.plot(peaks[0], unit[peaks[0]], 'r*')
            plt.plot(peaks[1], unit[peaks[1]],  'r*')
            plt.axhline(baseline, ls='--')

            plt.annotate("",
                     xy=(peaks[1], unit[peaks[0]]), xycoords='data',
                     xytext=(peaks[1], unit[peaks[1]]), textcoords='data',
                     arrowprops=dict(arrowstyle="<->",
                                     connectionstyle="arc3"),)
            plt.annotate("",
                     xy=(peaks[1], unit[peaks[0]]), xycoords='data',
                     xytext=(peaks[0], unit[peaks[0]]), textcoords='data',
                     arrowprops=dict(arrowstyle="<->",
                                     connectionstyle="arc3"),)
            plt.xticks(np.arange(0,49,5), np.round(np.arange(0,1.6,.165),2))
            plt.xlabel('Time (mS)')
            plt.ylabel('Amplitude ('+u"\u03bcV"+')')
            
            plt.text(peaks[0], unit[peaks[1]], "Duration " + str(np.round(duration,2)))
            plt.text(peaks[1], unit[peaks[0]], "Peak-trough ratio " + str(round(ratio,2)), rotation='vertical')
            plt.text(1, baseline+6, "Baseline", fontsize = 18)
            plt.text(peaks[1], unit[peaks[1]], "Peak")
            plt.text(peaks[0], unit[peaks[0]], "Trough")
            print(f"{j} group")
            print(f"{ratio} Peak-trough ratio", f"{amplitude} amplitude", f"{duration} peak-to-peak duration")
            
            
def draw_classified_waveforms(data_frame, units_mean_waveform_cleaned,plot_mean_waveforms=True, plot_violin =True):
    
    waveforms_per_label = []
    durations_label = []
    for i in list(set(data_frame['Labels'])):
        waveform_per_label = []
        duration_label = []
        plt.figure()
        plt.grid(b=None)
        for j in range(len(data_frame)):
            if data_frame['Labels'][j] == i:
               waveform_per_label.append(units_mean_waveform_cleaned[j])
               duration_label.append(data_frame['Durations'].values[j])
               plt.plot(units_mean_waveform_cleaned[j])
               plt.xticks(np.arange(0,49,5), np.round(np.arange(0,1.6,.165),2))
        durations_label.append(duration_label)
        waveforms_per_label.append(waveform_per_label)
    
    waveforms_0_label = []
    for array in waveforms_per_label[0]:
        waveforms_0_label.append(array)
    
    waveforms_0_label_array = np.array(waveforms_0_label)
    
    waveforms_1_label = []
    for array in waveforms_per_label[1]:
        waveforms_1_label.append(array)
    
    waveforms_1_label_array = np.array(waveforms_1_label)
    
    waveforms_2_label = []
    for array in waveforms_per_label[2]:
        waveforms_2_label.append(array)
    
    waveforms_2_label_array = np.array(waveforms_2_label)
    
    if plot_mean_waveforms:
        
        plt.figure()
        plt.title('Units waveforms', fontsize =22)
        plt.xlabel('Time(ms)', fontsize=18)
        
        plt.plot(waveforms_0_label_array.mean(axis=0),  color = 'purple', label = 'Regular Spikes')
        plt.xticks(np.arange(0,49,5), np.round(np.arange(0,1.6,.165),2))
        plt.fill_between( np.arange(0,50),waveforms_0_label_array.mean(axis=0) - waveforms_0_label_array.std(axis=0), waveforms_0_label_array.mean(axis=0) + waveforms_0_label_array.std(axis=0), color = 'purple', alpha=.4)
        
        plt.plot(waveforms_1_label_array.mean(axis=0), color = 'green', label = 'Fast Spikes')
        plt.xticks(np.arange(0,49,5), np.round(np.arange(0,1.6,.165),2))
        plt.fill_between( np.arange(0,50),waveforms_1_label_array.mean(axis=0) - waveforms_1_label_array.std(axis=0), waveforms_1_label_array.mean(axis=0) + waveforms_1_label_array.std(axis=0),color = 'green', alpha=.4)
        
        
        plt.plot(waveforms_2_label_array.mean(axis=0), color = 'blue', label = 'ni idea Spikes')
        plt.xticks(np.arange(0,49,5), np.round(np.arange(0,1.6,.165),2))
        plt.fill_between( np.arange(0,50),waveforms_2_label_array.mean(axis=0) - waveforms_2_label_array.std(axis=0), waveforms_2_label_array.mean(axis=0) + waveforms_2_label_array.std(axis=0),color = 'blue', alpha=.4)
        
        plt.legend()
        
    if plot_violin:
        plt.figure()
        plt.violinplot(durations_label, widths = 1.0, showmeans = True) 
        plt.xticks(np.arange(1,3.1,1))
        plt.xlabel('Labels', fontsize=18)
        plt.title('Durations density per label', fontsize =22)
        plt.ylabel('Durations(ms)', fontsize=18)
        plt.grid(b=None)
        
        plt.title('Durations density per label', fontsize =22)
        plt.ylabel('Durations(ms)', fontsize=18)