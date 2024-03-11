#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: %(Val-Calvo, Mikel and Alegre-Cortés, Javier)
@emails: %(mikel1982mail@gmail.com, jalegre@umh.es)
@institutions: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED), Postdoctoral Researcher Instituto de Neurociencias UMH-CSIC)
"""
#%%
import umap
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain
import similaritymeasures as sm


class Sorter:
    """
    A class for sorting spike data using dimensionality reduction and template matching.

    This class employs UMAP for reducing the dimensionality of the spike data and
    uses dynamic time warping for template matching.

    Methods:
    sort_spikes: Sorts the given spike data.
    _template_matching: Matches spike data to preloaded templates.
    _detect_similarUnits: Detects similar units in the spike data.
    _merge_similarClusters: Merges similar clusters based on detected similarities.
    """
    def __init__(self):
        path = './spikes_cleaning/libraries/spike_templates/'
        self.templates = np.array([np.load(path + 'spike_template_' + str(i) + '.npy') for i in range(115)])
        
    def sort_spikes(self, spikes: np.ndarray, n_neighbors: int = 20, min_dist: float = 0.3, n_components: int = 2,
                    metric: str = 'manhattan') -> np.ndarray:
        """
        Sorts the given spike data using UMAP and community detection.

        Parameters:
        spikes (np.ndarray): The array of spike data to be sorted.
        n_neighbors (int): The number of neighbors in UMAP.
        min_dist (float): The minimum distance in UMAP.
        n_components (int): The number of components for dimensionality reduction.
        metric (str): The metric used in UMAP.

        Returns:
        np.ndarray: Array of unit IDs after sorting.
        """
        if spikes.shape[0] <= n_neighbors:
            unit_IDs = np.zeros((spikes.shape[0],), dtype=int) + 1
        else:
            # scaling, maybe not necessary????¿?¿?¿?¿?
            spikes_ = spikes[:, 10:45]
            min_, max_ = spikes_.min(), spikes_.max()
            spikes_norm = np.array([(spk - min_)/(max_ - min_) for spk in spikes_])
            # compute latent features
            reducer = umap.UMAP( n_neighbors=min([n_neighbors,int(np.ceil(len(spikes_)/n_neighbors))]), min_dist=min_dist, n_components=n_components, metric=metric )
            reducer.fit_transform(spikes_norm)
            # compute the optimal set of clusters
            embedding_graph = nx.Graph(reducer.graph_)
            partition = community_louvain.best_partition(embedding_graph, resolution=1.1)
            unit_IDs = np.array([data[1]+1 for data in list(partition.items())])
           
            # revisite clusters
            if len(np.unique(unit_IDs)) > 1:
                unit_IDs = self._template_matching(unit_IDs, spikes)
                mylist = self._detect_similar_units(unit_IDs, spikes_norm, threshold=.7)
                unit_IDs = self._merge_similar_clusters(mylist, unit_IDs)
            
        return unit_IDs
    
    def _template_matching(self, unit_IDs: np.ndarray, spikes: np.ndarray) -> list:
        """
        Performs template matching on the given spike data.

        Parameters:
        unit_IDs (np.ndarray): The initial unit IDs for the spikes.
        spikes (np.ndarray): The spike data.

        Returns:
        list: Updated unit IDs after template matching.
        """
        
        for unit in np.unique(unit_IDs):
            unit_group = spikes[unit_IDs == unit]
            unit_group_mean = unit_group.mean(axis=0)
            
            max_similarity = np.inf
            which = -1
            for it, template in enumerate(self.templates):
                unit_group_mean = unit_group_mean[:len(template)] if len(template) < len(unit_group_mean) else unit_group_mean
                template_phase = np.vstack((template[:-1], np.diff(template)))
                unit_group_mean_phase = np.vstack((unit_group_mean[:-1], np.diff(unit_group_mean)))
                
                # -- check similarity ---
                similarity = sm.dtw(unit_group_mean_phase, template_phase)[0]
                if max_similarity > similarity:
                    max_similarity = similarity
                    which = it
                    
            # print(max_similarity, which)
            unit_IDs[unit_IDs == unit] = which
            # print(np.unique(unit_IDs))
            IDs = np.arange(len(np.unique(unit_IDs)))
            final_unit_IDs = [IDs[list(np.unique(unit_IDs)).index(u)] for u in unit_IDs]
        return final_unit_IDs
        
    @staticmethod
    def _detect_similar_units(units: np.ndarray, spikes: np.ndarray, threshold: float = 0.8) -> list:
        """
        Detects similar units in the spike data.

        Parameters:
        units (np.ndarray): The unit IDs of the spikes.
        spikes (np.ndarray): The spike data.
        threshold (float): The threshold for detecting similarity.

        Returns:
        list: A list of lists, each containing indices of similar units.
        """
        means = []
        num_spikesXcluster = []
        for label in np.unique(units):
            positions = [idx for idx,unit in enumerate(units) if unit == label]
            num_spikesXcluster.append(len(positions))
            means.append( spikes[positions].mean(axis=0) )

        mylist = []
        my_index = list(range(len(means)))
        
        aux_means = means[1:]
        aux_my_index = my_index[1:]
        
        index = 0
        for it in range(len(means)):
            main_mean = means[index]

            equal, distinct = [],[]
            for aux, idx in zip(aux_means,aux_my_index):
                aux_phase = np.vstack((aux[:-1], np.diff(aux)))
                main_mean_phase = np.vstack((main_mean[:-1], np.diff(main_mean)))

                # -- check similarity ---
                similarity = sm.dtw(main_mean_phase, aux_phase)[0]
                if similarity < (1/np.mean(num_spikesXcluster))+threshold:
                    equal.append(idx)
                else:
                    distinct.append(idx)
            
            if equal:
                equal.append(index)
                mylist.append(equal)
            elif not equal and len(distinct) >= 1:
                mylist.append([my_index[index]])
            
            if not distinct:
                break
            elif len(distinct) == 1:
                mylist.append(distinct)
                break
            else:
                aux_means = [means[val] for val in distinct[1:]]
                aux_myindex = [my_index[val] for val in distinct[1:]]
                index = distinct[0]
                
        return mylist

    @staticmethod
    def _merge_similar_clusters(mylist: list, units: np.ndarray) -> np.ndarray:
        """
        Merges similar clusters in the spike data.

        Parameters:
        mylist (list): A list of lists with indices of similar units.
        units (np.ndarray): The unit IDs of the spikes.
        spikes (np.ndarray): The spike data.

        Returns:
        np.ndarray: Updated unit IDs after merging similar clusters.
        """

        print(type(units), units)
        units = np.array(units)
        labels = np.unique(units)

        for idx, sublist in enumerate(mylist):
            for position in sublist:
                index = np.array([pos for pos, unit in enumerate(units) if unit == labels[position]], dtype=np.int16)
                print(index)
                for it in index:
                    units[it]  = (idx+1)*-1

        return abs(units)
            
        
