import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter(fig, x, y, z):
    cmap = plt.get_cmap('viridis_r')
    cmap.set_under('red')
    scat = fig.scatter(x=x, y=y, c=z, cmap=cmap)
    fig.set_alpha(0.5)
    plt.colorbar(scat, extend='min')
    return fig

def plot_dist(fig, data_arr, labels, title, cumulative=False, log_scale=False):
    if len(data_arr) != len(labels):
        return "Number of datasets does not match number of labels"
    cmap = plt.get_cmap('turbo', len(data_arr))
    for i in range(len(data_arr)):
        sns.kdeplot(data=data_arr[i], color=cmap(i), label=labels[i], fill=True, ax=fig, cut=0, cumulative=cumulative, 
                    common_norm=True, common_grid=True, log_scale=log_scale)
    fig.legend()
    fig.set_title(title)
    return fig

def plot_heatmap(fig, x, y, z):
    X,Y = np.meshgrid(x,y)
    Z=z.reshape(len(x),len(y))
    Z=np.transpose(Z)
    hmap = fig.imshow(Z , cmap = 'jet' , interpolation = 'gaussian' , 
           origin='lower', aspect='equal',  extent = [min(x), max(x), min(y), max(y)])
    plt.colorbar(hmap, extend='min')
    return fig