#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: %(Val-Calvo, Mikel and Alegre-Cortés, Javier)
@emails: %(mikel1982mail@gmail.com, jalegre@umh.es)
@institutions: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED), Postdoctoral Researcher Instituto de Neurociencias UMH-CSIC)
"""
#%%
import numpy as np
import umap
import networkx as nx
import community.community_louvain as community_louvain
import similaritymeasures
from scipy.stats import spearmanr
from scipy.stats import zscore
from scipy import signal
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib


class SpikeDenoiser:
    """
    A class for denoising spike data using signal processing and machine learning techniques.

    Methods:
    show_spike_templates: Visualize spike templates.
    show_artefact_templates: Visualize artefact templates.
    denoise: Perform denoising on waveforms.
    _remove_small_clusters: Remove small clusters of spikes.
    _filter_spikes: Filter out spikes based on various criteria.
    """
    def __init__(self):
        """
        Initializes the spike denoiser with default size threshold.
        """
        self.size_threshold = 1000
    
    def __load_references(self) -> bool:
        """
        Private method to load spike and artefact templates from predefined directories.

        Returns:
        bool: Always returns True after loading templates.
        """
        mypath = './spikes_cleaning/libraries/spike_templates/'
        self.references = []
        for file in [f for f in listdir(mypath) if isfile(join(mypath, f))]:
            self.references.append( np.load(mypath + file) )
            
        mypath = './spikes_cleaning/libraries/artefact_templates/'
        self.antireferences = []
        for file in [f for f in listdir(mypath) if isfile(join(mypath, f))]:
            self.antireferences.append( np.load(mypath + file) )
            
        return True

    @staticmethod
    def show_spike_templates() -> None:
        """
        Visualize and save spike templates as images.
        """
        mypath = './spikes_cleaning/libraries/spike_templates/'
        files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        
        for file in files:
            plt.figure()  
            plt.plot(np.load(mypath+file), 'c', linewidth=4)
            plt.title(file[:-4])

    @staticmethod
    def show_artefact_templates() -> None:
        """
        Visualize and save artefact templates as images.
        """
        mypath = './spikes_cleaning/libraries/artefact_templates/'
        files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        
        for file in files:
            plt.figure()  
            plt.plot(np.load(mypath+file), 'c', linewidth=4)
            plt.title(file[:-4])

    def denoise(self, waveforms: np.ndarray, n_neighbors: int = 10, min_dist: float = 0.3, n_components: int = 2, metric: str = 'manhattan') -> np.ndarray:
        """
        Perform denoising on given waveforms using UMAP and community detection.

        Parameters:
        waveforms (np.ndarray): The array of waveforms to be denoised.
        n_neighbors (int): The number of neighbors in UMAP.
        min_dist (float): The minimum distance in UMAP.
        n_components (int): The number of components for dimensionality reduction.
        metric (str): The metric used in UMAP.

        Returns:
        np.ndarray: Array of unit IDs after denoising.
        """
        self.__load_references()
        if len(waveforms) <= n_neighbors:
            unit_IDs = np.array([-1]*len(waveforms))#self._filter_spikes(waveforms, np.zeros((len(waveforms),), dtype=int))#
        else:
            print('que esta pasando en denoise ', waveforms.shape, waveforms)
            min_, max_ = waveforms.min(), waveforms.max()
            waveforms_norm = np.array([(wave - min_)/(max_ - min_) for wave in waveforms], dtype=np.float64)
            reducer = umap.UMAP( n_neighbors=n_neighbors, min_dist=min_dist, 
                                n_components=n_components, metric=metric, 
                                set_op_mix_ratio=0.2 )
            reducer.fit_transform(waveforms_norm)
            embedding_graph = nx.Graph(reducer.graph_)
            partition = community_louvain.best_partition(embedding_graph)
            labels = np.array([data[1] for data in list(partition.items())])
            
            unit_IDs = self._filter_spikes(waveforms_norm, labels)
            unit_IDs = self._remove_small_clusters( unit_IDs )
            
        return unit_IDs

    @staticmethod
    def _remove_small_clusters(unit_IDs: np.ndarray, min_spikes_required: int = 20) -> np.ndarray:
        """
        Remove small clusters of spikes that don't meet the minimum spike count.

        Parameters:
        unit_IDs (np.ndarray): Array of unit IDs.
        min_spikes_required (int): Minimum number of spikes required to retain a cluster.

        Returns:
        np.ndarray: Updated array of unit IDs.
        """
        if len([ID for ID in unit_IDs if ID == 1]) < min_spikes_required:
            unit_IDs = [-1 for ID in unit_IDs ]
            
        return unit_IDs
        
    def _filter_spikes(self, waveforms: np.ndarray, labels: np.ndarray, plot: bool=False) -> np.ndarray:
        """
        Filter out spikes based on correlation with references and other criteria.

        Parameters:
        waveforms (np.ndarray): The waveforms to be filtered.
        labels (np.ndarray): The labels associated with each waveform.

        Returns:
        np.ndarray: Array of unit IDs after filtering.
        """
        unit_IDs = np.zeros_like(labels)              
                
        for label in np.unique(labels):
            # -- select units from one cluster
            index = [i for i,x in enumerate(labels==label) if x]
            # ---  compute the mean of the cluster ----------------
            y = zscore(waveforms[index].mean(axis=0))

            spk_ccorr = -np.inf
            for it, reference in enumerate(self.references):
                min_ = min(len(y),len(reference))
                y = y[:min_]
                reference = reference[:min_]
                    
                corr, _ = spearmanr(zscore(reference), y)
                if spk_ccorr < corr:
                    spk_ccorr = corr
                    final_reference = zscore(reference)
   
            is_noise = False
            for it, antireference in enumerate(self.antireferences):
                min_ = min(len(y),len(antireference))
                y = y[:min_]
                antireference = antireference[:min_]
                
                corr, _ = spearmanr(zscore(antireference), y)

                if spk_ccorr < corr and corr > .9:
                    is_noise = True
                    final_antireference = zscore(antireference)
                    
            z = np.polyfit(np.arange(1,len(y)+1), y, 1)
            p = np.poly1d(z)

            final_reference_phase = np.vstack((final_reference[:-1], np.diff(final_reference)))
            y_phase = np.vstack((y[:-1], np.diff(y)))

            # -- check spike alignment --
            corr = signal.correlate(final_reference, y)
            desplazamiento = int(np.argmax(corr) - corr.size/2)
            # print('desplazamiento ', abs(desplazamiento))
            if abs(desplazamiento) < 5:
                if desplazamiento < 0:
                    final_reference = final_reference[:desplazamiento]
                    y = y[abs(desplazamiento):]
                    final_reference_phase = np.vstack((final_reference[:-1], np.diff(final_reference)))
                    y_phase = np.vstack((y[:-1], np.diff(y)))
                elif desplazamiento > 0:
                    final_reference = final_reference[desplazamiento:]
                    y = y[:-desplazamiento]
                    final_reference_phase = np.vstack((final_reference[:-1], np.diff(final_reference)))
                    y_phase = np.vstack((y[:-1], np.diff(y)))
                        
            if y_phase.shape[1] != final_reference_phase.shape[1]:
                min_len = min(y_phase.shape[1], final_reference_phase.shape[1])
                y_phase = y_phase[:,:min_len]
                final_reference_phase = final_reference_phase[:,:min_len]
            print(y_phase.shape, final_reference_phase.shape)
            df = similaritymeasures.dtw(final_reference_phase, y_phase)[0]
            
            std = np.sum(np.std(zscore(waveforms[index], axis=1), axis=0))/50

            if not is_noise and spk_ccorr > .7 and p[1]*100 > -2 and p[1]*100 < 4 and df < 6 and std <= .6: #
                unit_IDs[index] = 1

            else:
                unit_IDs[index] = -1

            if plot:
                plt.figure()
                plt.plot(final_antireference, 'r')
                plt.plot(final_reference, 'g')
                plt.plot(y, 'b')
                plt.title('corr with noise: ' + str(corr) + ' corr with spikes ' + str(spk_ccorr))

                plt.figure()
                plt.subplot(211)
                plt.plot(y)
                plt.plot(np.arange(1,49)*p[1]+p[0], 'm')
                if is_noise:
                    plt.plot(final_antireference, 'r')
                plt.plot(final_reference, 'g')
                plt.subplot(212)

                if not is_noise and spk_ccorr > .7 and p[1] * 100 > -2 and p[1] * 100 < 4 and df < 6 and std <= .6:  #
                    for wave in waveforms[index]:
                        plt.plot(zscore(wave), 'g')
                else:
                    for wave in waveforms[index]:
                        plt.plot(zscore(wave), 'r')
                plt.suptitle('result ' + str(not is_noise and spk_ccorr > .7 and p[1]*100 > -2 and p[1]*100 < 4 and df < 6 and std < .6) + " isnoise " + str(is_noise) + "corr {:.2f}".format(spk_ccorr) + "p {:.2f}".format(p[1]*100) + "df {:.2f}".format(df)+"std {:.2f}".format(std))

        return unit_IDs
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    