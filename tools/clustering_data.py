# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:13:43 2024

@author: Rocio Lopez Peco
@email: rocio.lopezp@umh.es
"""

#reduce data in two dimensions


import umap
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.patches import Ellipse
import numpy as np



def UMAP_reducer(new_units_all, plot_UMAP = True):

    reducer = umap.UMAP(random_state=10) #elegir un valor adecuado que dibuje siempre igual random_state=41
    
    embedding = reducer.fit_transform(new_units_all)
    embedding.shape
    
    if plot_UMAP:
        plt.figure()
        plt.scatter(embedding[:287, 0],embedding[:287, 1], c = 'orange', label = 'Miguel units')
        plt.scatter(embedding[288:, 0],embedding[288:, 1], c = 'green', label = 'Berna units')
        plt.legend()
        plt.grid(b=None)
        plt.title('Subjects units representation', fontsize = 22)
        
    return embedding

def Silhouette_visualizer(embedding):
    
    fig, ax = plt.subplots(2, 2, figsize=(15,8))
    plt.suptitle('Silhouette Analysis for 2,3,4,5 clusters')

    for i in [2, 3, 4, 5]:
        '''
        Create KMeans instance for different number of clusters
        '''
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        q, mod = divmod(i, 2)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        '''
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(embedding)
    

        
def GMM_classifier(embedding, plot_clustering =True):
    gmm = GMM(n_components=3, covariance_type="full", random_state=None).fit(embedding)
    labels = gmm.predict(embedding)
    probs = gmm.predict_proba(embedding)
    print(probs[:5].round(3))

    if plot_clustering: 
        plt.figure()
        plt.legend(['Group 1' , 'Group 2', 'Group 3'])
        plt.grid(b=None)
    
        plt.title('Units clustering', fontsize = 22)
        size = 50 * probs.max(1) ** 2  # square emphasizes differences
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='PiYG', s=size, alpha = 0.6, );
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=40, cmap='PiYG', zorder=2,  alpha = 0.6, )
        plt.axis('equal')
        w_factor = 0.2 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)

    return labels

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        

        
       
        
        