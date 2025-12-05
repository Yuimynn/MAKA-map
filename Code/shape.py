'''
Visualize the distance matrix between protein residues, and present the.npy format distance data intuitively through a heat map.
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.pyplot import figure

# File path
npy_file_path = 'Predicted npy file path'
output_path = 'Output folder'


distance_matrix = np.load(npy_file_path)
# Visual function
def plot_protein_io(X, Y, save_path=None):
    figure(num=None, figsize=(20, 54), dpi=80, facecolor='w', frameon=True, edgecolor='k')
    print('')
    print('Generating seaborn plots.. patience..')
    for i in range(0, len(X[0, 0, :])):
        plt.subplot(14, 4, i + 1)
        sns.heatmap(X[:, :, i], cmap='RdYlBu')
        plt.title('Channel ' + str(i))
    plt.subplot(14, 4, len(X[0, 0, :]) + 1)
    plt.grid(None)
    y = np.copy(Y)
    y[y > 25.0] = 25.0
    sns.heatmap(y, cmap='Spectral')
    plt.title('True Distances')

    # Save the image in the .png format
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Image saved to {save_path}')
    else:
        plt.show()

X = np.expand_dims(distance_matrix, axis=2) 
Y = distance_matrix
plot_protein_io(X, Y, os.path.join(output_path, 'shape.png'))




