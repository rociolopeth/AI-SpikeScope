from abc import ABC, abstractmethod
from .brpylib import NsxFile, NevFile
from .UTAH_array import UTAH
from .filters import butter_highpass
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np

from typing import Any, Dict, List, Union, Optional

class DataLoader(ABC):
    @abstractmethod
    def load_data(self, file_path) -> Dict:
        """
        Load data from a file.

        :param file_path: Path to the data file
        """
        pass


class NSXLoader(DataLoader):
    def load_data(self, file_path) -> Dict:
        """
            Load and process data from an NSX file, optionally plotting the data.

            This function reads data from a specified NSX file, reorders it based on electrode mappings,
            and optionally plots the z-score normalized signals for each electrode.

            Parameters:
            file_path (str): The path to the NSX file.
            plot (bool): A flag to indicate whether to plot the data or not.

            Returns:
            Dict[str, Any]: A dictionary containing the processed data.
            """

        nsx_file = NsxFile(file_path)
        data_container = nsx_file.getdata()
        nsx_file.close()
        del nsx_file

        # Mapping electrodes to channels and vice versa
        utah = UTAH()

        # Reordering data based on electrode configuration
        raw = data_container['data'][:96]
        ordered_raw = np.zeros_like(raw)
        for channel, signal in enumerate(raw):
            electrode = utah.channel2electrode(channel + 1)
            ordered_raw[int(electrode) - 1] = raw[channel]
        data_container['data'] = ordered_raw

        # Visualize the data if plot is True
        if plot:
            plt.figure()
            for it, wave in enumerate(data_container['data']):
                if it < 97:
                    plt.plot(zscore(wave) + it * 5)
            plt.yticks(np.arange(1, 97) * 5,
                       ['E' + str(it) + ' CH' + str(utah.electrode2channel[str(it)]) for it in np.arange(1, 97)])

        return data_container

    @staticmethod
    def get_spikes_from_raw( raw_container: Dict[str, np.ndarray], triggers: Optional[List[int]] = None,
                             path: str = '', fs_raw: int = 30000, low: int = 250, std: int = 5,
                             step: int = 12, length: int = 50) -> Dict[str, List]:
        """
        Extract spike data from raw neural signal data.

        This function applies high-pass filtering and thresholding to raw signal data to identify spikes.
        It then extracts and returns these spikes in a structured format.

        Parameters:
        raw_container (Dict[str, np.ndarray]): The container with raw neural signal data.
        triggers (Optional[List[int]]): Optional list of trigger points for spike extraction.
        path (str): The file path of the raw data.
        fs_raw (int): Sampling rate of the raw data.
        low (int): Low frequency threshold for high-pass filtering.
        std (int): Standard deviation multiplier for spike detection threshold.
        step (int): Step size used in spike extraction.

        Returns:
        Dict[str, List]: A dictionary containing extracted spike data.
        """

        spike_dict = {
            'ElectrodeID': [], 'FileNames': path, 'SamplingRate': fs_raw, 'ExperimentID': [],
            'Active': [], 'ChannelID': [], 'UnitID': [], 'OldID': [], 'TimeStamps': [],
            'Waveforms': [], 'Triggers': [np.array(triggers)] if triggers else [],
            'Triggers_active': [True] if triggers else []
        }

        spike_dict['Triggers'] = [np.array(triggers)] if triggers else []
        spike_dict['Triggers_active'] = [True] if triggers else []
        spike_dict['FileNames'] = path
        spike_dict['SamplingRate'] = fs_raw

        append_ElectrodeID = spike_dict['ElectrodeID'].append
        append_ChannelID = spike_dict['ChannelID'].append
        append_TimeStamps = spike_dict['TimeStamps'].append
        append_Waveforms = spike_dict['Waveforms'].append
        append_UnitID = spike_dict['UnitID'].append
        append_ExperimentID = spike_dict['ExperimentID'].append
        append_Active = spike_dict['Active'].append
        append_OldID = spike_dict['OldID'].append

        utah = UTAH()

        for electrode in range(1, 97):
            raw = raw_container['data'][electrode - 1]
            if triggers:
                threshold = butter_highpass(raw[triggers[0] - fs_raw:triggers[0]], low, fs_raw)
            else:
                threshold = butter_highpass(raw, low, fs_raw)

            filtered = butter_highpass(raw, low, fs_raw)

            threshold = threshold.std() * std

            print(' electrode ', electrode, ' applyied threshold_std ', std, ' signal std*threshold_std ',
                  filtered.std() * std, 'threshold ', threshold)
            x = filtered < -threshold
            transitions = ~x[:-1] & x[1:]
            index = [it for it, val in enumerate(transitions) if val]

            for it, value in enumerate(index):
                if value >= step and len(filtered[value - step:value + abs(length - step)]) == length:
                    append_Waveforms(filtered[value - step:value + abs(length - step)])
                    append_ElectrodeID(electrode)
                    append_ChannelID(utah.electrode2channel[str(electrode)])
                    append_TimeStamps(value)
                    append_UnitID(1)
                    append_ExperimentID(0)
                    append_Active(True)
                    append_OldID(None)

        return spike_dict

class TriggerLoader(DataLoader):
    def load_data(self, file_path) -> Dict:
        """
            Load and process trigger data from a NEV file.

            This function opens a NEV file, extracts spike event data, and processes trigger information.
            It calculates the unique trigger IDs and their corresponding timestamps, identifying the first
            trigger in each sequence.

            Parameters:
            file_path (str): The path to the NEV file.

            Returns:
            Dict: A dictionary with trigger IDs as keys and a nested dictionary
            containing lists of all trigger timestamps ('All') and the first timestamp in the sequence ('First').
            """

        nev_file = NevFile(file_path)
        data_container = nev_file.getdata('all')
        nev_file.close()
        del nev_file

        # Extracting trigger IDs from spike events
        triggers_ID = [ch - 5120 for ch in np.unique(data_container['spike_events']['Channel']) if ch > 128]

        triggers = {ID: {'First': [], 'All': []} for ID in triggers_ID}

        for key in triggers:
            index = [it for it, ch in enumerate(data_container['spike_events']['Channel']) if ch - 5120 == key]
            triggers[key]['All'] = list(np.array(data_container['spike_events']['TimeStamps'])[index])

            # Identifying the first trigger in each sequence
            check = list(np.diff(triggers[key]['All']) > 50)

            if np.any(check) and not np.all(check):
                index = [0] + [it + 1 for it, val in enumerate(check) if val]
                triggers[key]['First'] = [triggers[key]['All'][it] for it in index]
            elif np.all(check):
                triggers[key]['First'] = triggers[key]['All']
            else:
                triggers[key]['First'] = [triggers[key]['All'][0]]

        return triggers


def get_loader(file_type):
    if file_type == "nsx":
        return NSXLoader()
    elif file_type == "trigger":
        return TriggerLoader()
    else:
        raise ValueError("Unsupported file type")


def load_and_process_file(file_path, file_type):
    loader = get_loader(file_type)
    data = loader.load_data(file_path)

    # Check if the loader has the `get_spikes_from_raw` method
    if hasattr(loader, 'get_spikes_from_raw'):
        spikes = loader.get_spikes_from_raw(data)
        return spikes
    else:
        return data



