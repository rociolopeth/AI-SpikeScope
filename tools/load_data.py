# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:04:49 2024

@authors: Rocio Lopez Peco, Mikel Val Calvo
@email: yrociro@gmail.es, mikel1982mail@gmail.com
"""

import numpy as np 
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.stats import zscore


def get_template( file, plot=True ):
    template_dict = np.load(file, allow_pickle=True).item()
    template = template_dict['waveform']
    
    if plot:
        plt.figure()
        plt.plot(template)
        plt.title( 'Channel: ' + str(template_dict['ChannelID']) + ' Unit ' + str(template_dict['UnitID']) + ' num spikes ' +  str(template_dict['num_spikes']))
    return template_dict


def extract_data(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    units_A = []
    units_B = []

    for it,file in enumerate(onlyfiles):
        template_dict = get_template(join(mypath, file), False)
        if template_dict['subject']== 'berna':
            units_A.append( template_dict['waveform'] )
        elif template_dict['subject']== 'miguel':
            units_B.append( template_dict['waveform'])
    
    new_units_A = []
    for wave in units_A:
        if len(wave) == 50:
            new_units_A.append(zscore(wave)[:-2] )
        else:
            new_units_A.append(zscore(wave))
        
        
    new_units_B = []
    for wave in units_B:
        if len(wave) == 50:
            new_units_B.append(zscore(wave)[:-2] )
    
        else:
            new_units_B.append(zscore(wave))
    
    new_units_all =[]
    
    for value in new_units_B:
      new_units_all.append(value)
    for value in new_units_A:
      new_units_all.append(value)
      
    print('len(new_units_all): ', len(new_units_all))
    return new_units_all


def load_auto_cleaned_data(mypath_processed):
    #load files and extract units waveforms to work them 
    units_mean_waveform = []
    onlyfilesprocessed = [f for f in listdir(mypath_processed) if isfile(join(mypath_processed, f))]
    
    for f in onlyfilesprocessed:
        file = np.load(mypath_processed+f, allow_pickle=True).item()    
        file_dict = {'Channel': file['ChannelID'],
                      'Unit': file['UnitID'],
                      'Waveforms': file['Waveforms']}
                
    
        for channel in np.unique(file_dict['Channel']):
            channel_index = [it for it,ch in enumerate(file_dict['Channel']) if ch == channel]
            units = np.unique( np.array(file_dict['Unit'])[channel_index] )
            
            for unit in units:
                if unit != -1:
                    unit_index = [it for it in channel_index if file_dict['Unit'][it] == unit]
                    unit_mean_wave = np.mean( np.array(file_dict['Waveforms'])[unit_index], axis=0)
                    units_mean_waveform.append(zscore(unit_mean_wave))
                    
    return units_mean_waveform


