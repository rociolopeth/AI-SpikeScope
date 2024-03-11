#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: %(Val-Calvo, Mikel and Alegre-Cortés, Javier)
@emails: %(mikel1982mail@gmail.com, jalegre@umh.es)
@institutions: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED), Postdoctoral Researcher Instituto de Neurociencias UMH-CSIC)
"""

import matplotlib.pyplot as plt
from spikes_cleaning.libraries.LOAD_ripple_datafiles import load_nsx, load_triggers, get_spikes_from_raw
from spikes_cleaning.libraries.SPIKES_denoising_strategies_multiprocessing import clean_and_sort
import numpy as np
from os import listdir
from os.path import isfile, join
from typing import List


def process_spikes_cleaning_and_sorting(file_path: str, save_results: bool = False) -> None:
    """
    Process the spikes by cleaning and sorting from the given file path.

    This function iterates over files in the specified directory, loads the raw data,
    extracts triggers, processes spikes, and finally saves the cleaned and sorted spikes.

    Parameters:
    file_path (str): The path to the directory containing the raw .ns5 and .nev files.

    Returns:
    None: This function does not return anything but saves processed spike data to disk.
    """

    # Retrieve file paths for processing
    paths: List[str] = [f[:-4] for f in listdir(file_path) if isfile(join(file_path, f)) and f[-4:] == '.ns5']

    for path in paths:
        print(f"Processing file: {path}")
        # Load raw data
        raw_container = load_nsx(file_path + path + '.ns5', plot=False)

        # Load triggers
        triggers_dict = load_triggers(file_path + path + '.nev')

        if triggers_dict:
            # Assumes single channel stimulation or simultaneous multi-channel stimulation
            triggers = [triggers_dict[key]['First'] for key in triggers_dict][0]
        else:
            triggers = None

        # Extract spikes from raw data
        spike_dict = get_spikes_from_raw(raw_container, triggers)

        spike_dict = np.load('./dataset/spike_dict_testing.npy', allow_pickle=True).item()
        print(spike_dict.keys())

        # Clean and sort spikes
        spike_dict_cleaned = clean_and_sort(spike_dict)

        # Save the cleaned and sorted spike dictionary
        if save_results:
            np.save(file_path + path[:-4] + '_spike_dict_cleaned.npy', spike_dict_cleaned, allow_pickle=True)

        # Memory management
        del raw_container, triggers, spike_dict, spike_dict_cleaned



