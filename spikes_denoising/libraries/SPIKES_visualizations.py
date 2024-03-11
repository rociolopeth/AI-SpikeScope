#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: %(Val-Calvo, Mikel and Alegre-Cortés, Javier)
@emails: %(mikel1982mail@gmail.com, jalegre@umh.es)
@institutions: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED), Postdoctoral Researcher Instituto de Neurociencias UMH-CSIC)
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Union


def spikes_per_channel(spike_dict: Dict[str, List], channel: int, unit: Union[str, int] = 'All') -> None:
    """
    Plots spike waveforms for a specific channel, optionally filtered by unit identifier.

    This function visualizes the waveforms of spikes corresponding to a given channel.
    It allows for an optional filter to display waveforms for a specific unit.

    Parameters:
    spike_dict (Dict[str, List]): Dictionary containing spike data.
    channel (int): The channel number for which to plot the waveforms.
    unit (Union[str, int]): The unit identifier to filter the spikes. Use 'All' for no filter.

    Returns:
    None: This function does not return anything but creates a plot.
    """
    # Determine the index of spikes to plot based on channel and unit filters
    if unit == 'All':
        index = [it for it,ch in enumerate(spike_dict['ChannelID']) if ch == channel and spike_dict['UnitID'][it] != -1]
    else:
        index = [it for it,ch in enumerate(spike_dict['ChannelID']) if ch == channel and spike_dict['UnitID'][it] != -1 and spike_dict['UnitID'][it] == unit]

    # Plot the waveforms if any are found
    if len(index) > 0:
        cmap = plt.get_cmap('Set1') # Color map for different units

        plt.figure()
        plt.title('Channel '+str(channel))
        for it in index:
            plt.plot(spike_dict['Waveforms'][it], color=cmap(spike_dict['UnitID'][it]))


