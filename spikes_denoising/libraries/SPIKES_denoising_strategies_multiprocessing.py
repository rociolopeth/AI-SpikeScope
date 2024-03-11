# -*- coding: utf-8 -*-
"""
@authors: %(Val-Calvo, Mikel and Alegre-Cortés, Javier)
@emails: %(mikel1982mail@gmail.com, jalegre@umh.es)
@institutions: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED), Postdoctoral Researcher Instituto de Neurociencias UMH-CSIC)
"""

from .SPIKES_cleaner import SpikeDenoiser
from .SPIKES_sorter_umap_louvain_templates import Sorter
from .LOAD_ripple_datafiles import load_nsx, get_spikes_from_raw, load_triggers
from .SPIKES_visualizations import spikes_per_channel

import matplotlib.pyplot as plt
import numpy as np
import sys
import copy

def fully_automatic(channel_dict, r_min=-500, r_max=500, n_neighbors=10, min_dist=.3, metric='manhattan'):
    """
    Perform a full analysis pipeline on neural spike data, including amplitude threshold analysis,
    denoising, and sorting.

    Parameters:
    channel_dict (Dict[str, Any]): A dictionary containing channel data and metadata.
    r_min (int): The minimum amplitude threshold.
    r_max (int): The maximum amplitude threshold.
    n_neighbors (int): The number of neighbors for denoising.
    min_dist (float): The minimum distance for denoising.
    metric (str): The metric used for denoising.

    Returns:
    None: This function modifies the channel_dict in place and does not return any value.
    """
    ######### AMPLITUDE THRESHOLD ANALYSIS ############################################################
    print('STARTING AMPLITUDE THRESHOLD ANALYSIS ')
    index = np.arange(len(channel_dict['UnitID']))

    if len(index) > 0:
        # get the corresponding waveforms
        waveforms = np.array(channel_dict['Waveforms'])[index]
        sub_index = np.array(
            [idx for it, idx in enumerate(index) if (waveforms[it].min() < r_min or waveforms[it].max() > r_max)])
        if len(sub_index) > 0:
            unitIDs = np.array(channel_dict['UnitID'])
            unitIDs[sub_index] = -1
            channel_dict['UnitID'] = list(unitIDs)
            print('threshold analysis: ', channel_dict['UnitID'])
        ######### CLEANER ANALYSIS ############################################################
        print('STARTING CLEANER ANALYSIS ')
        denoise_index = np.array([it for it, unitID in enumerate(channel_dict['UnitID']) if unitID != -1])
        if len(denoise_index) > 0:
            print('######## waveforms algo pasa en denoise index ', denoise_index, '##############')
            waveforms = process_waveforms_for_denoising(channel_dict, denoise_index)
            print(waveforms.shape)

            spk = SpikeDenoiser()
            scores = spk.denoise(waveforms, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
            # marked as noise
            unitIDs = np.array(channel_dict['UnitID'])
            unitIDs[denoise_index] = scores
            channel_dict['UnitID'] = list(unitIDs)
            print('denoising analysis: ', channel_dict['UnitID'])
        ######### SORTING ANALYSIS ############################################################
        print('STARTING SORTING ANALYSIS ')
        sorting_index = np.array([it for it, unitID in enumerate(channel_dict['UnitID']) if unitID != -1])
        if len(sorting_index) > 0:
            waveforms = np.array(channel_dict['Waveforms'])[sorting_index]
            spike_sorter = Sorter()
            scores = spike_sorter.sort_spikes(waveforms)
            unitIDs = np.array(channel_dict['UnitID'])
            unitIDs[sorting_index] = scores
            channel_dict['UnitID'] = list(unitIDs)
            print('sorting analysis: ', channel_dict['UnitID'])



def process_waveforms_for_denoising(channel_dict, denoise_index):
    """
    Process waveforms for denoising analysis.

    Parameters:
    channel_dict (Dict[str, Any]): A dictionary containing channel data.
    denoise_index (List[int]): Indices of units to be denoised.

    Returns:
    np.ndarray: An array of processed waveforms for denoising.
    """
    waveforms = []
    for it in denoise_index:
        wave = channel_dict['Waveforms'][it]
        if len(wave) == 50:
            waveforms.append(wave)
        elif len(wave) < 50:
            wave = np.array(list(wave) + [0] * (50 - len(wave)))
            waveforms.append(wave)
    return np.array(waveforms)


def get_channelDict(channel_dict, spike_dict, channel):
    channel_dict['ElectrodeID'] = []
    channel_dict['ChannelID'] = []
    channel_dict['UnitID'] = []
    channel_dict['TimeStamps'] = []
    channel_dict['Waveforms'] = []

    index = [it for it, ch in enumerate(spike_dict['ChannelID']) if ch == channel]
    print('channel ', channel, 'len index ', len(index))
    if len(index) > 0:
        channel_dict['ElectrodeID'] = [spike_dict['ElectrodeID'][it] for it in index]
        channel_dict['ChannelID'] = [spike_dict['ChannelID'][it] for it in index]
        channel_dict['UnitID'] = [spike_dict['UnitID'][it] for it in index]
        channel_dict['TimeStamps'] = [spike_dict['TimeStamps'][it] for it in index]
        channel_dict['Waveforms'] = [spike_dict['Waveforms'][it] for it in index]

    print(len(channel_dict['ElectrodeID']))


# %%
def clean_and_sort(spike_dict):
    # Assuming get_channelDict and fully_automatic are defined elsewhere
    dictionaryOFchannels = {}

    # Initialize dictionary for each channel without multiprocessing Manager
    for channel in np.unique(spike_dict['ChannelID']):
        dictionaryOFchannels[channel] = {}
        get_channelDict(dictionaryOFchannels[channel], spike_dict, channel)

    # Sequentially process each channel without semaphores or multiprocessing
    for channel in np.unique(spike_dict['ChannelID']):
        print(f"Processing channel: {channel}")
        fully_automatic(dictionaryOFchannels[channel])

    print('Finish all the processing!!')
    # %% copy in the final dictionary the proxy dicts data
    final_spike_dict = copy.deepcopy(spike_dict)
    print(final_spike_dict['SamplingRate'], final_spike_dict['Triggers'], final_spike_dict['Triggers_active'])
    final_spike_dict['ElectrodeID'] = []
    final_spike_dict['ChannelID'] = []
    final_spike_dict['UnitID'] = []
    final_spike_dict['TimeStamps'] = []
    final_spike_dict['Waveforms'] = []
    final_spike_dict['OldID'] = []
    final_spike_dict['ExperimentID'] = []
    final_spike_dict['Active'] = []

    for channel in np.unique(spike_dict['ChannelID']):
        # final_spike_dict['ElectrodeID'] += dictionaryOFchannels[channel]['ElectrodeID']
        final_spike_dict['ChannelID'] += dictionaryOFchannels[channel]['ChannelID']
        final_spike_dict['UnitID'] += dictionaryOFchannels[channel]['UnitID']
        final_spike_dict['TimeStamps'] += dictionaryOFchannels[channel]['TimeStamps']
        final_spike_dict['Waveforms'] += dictionaryOFchannels[channel]['Waveforms']
        final_spike_dict['OldID'] += dictionaryOFchannels[channel]['UnitID']
        final_spike_dict['ExperimentID'] += [0] * len(final_spike_dict['ChannelID'])
        final_spike_dict['Active'] += [True] * len(final_spike_dict['ChannelID'])

    # %% clear the proxy dicts from ram memory
    for channel in np.unique(spike_dict['ChannelID']):
        dictionaryOFchannels[channel].clear()

    del dictionaryOFchannels
    del spike_dict

    return final_spike_dict

