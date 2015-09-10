#Clustering of parameter vectors

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import sys
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tsne import bh_sne
cm = plt.cm.get_cmap('RdYlBu')


#data = pickle.load(open(sys.argv[1]))
data = pickle.load(open('pickled_earm_traces_4_chains_100000_samples_0point1_std.p'))

# extract parameters and scores from pickeled traces
params = [ i for i in data['params']]
params = np.asarray(params)
params = params.reshape(np.shape(params)[0]*np.shape(params)[1],np.shape(params)[2])
print "shape of all data = ",len(params)
scores = [ i for i in data['error']]
scores = np.asarray(scores)
scores = scores.reshape(np.shape(scores)[0]*np.shape(scores)[1])
print "shape of all scores = ",np.shape(scores)


# threshold the data to be above 0
above_avg= np.where( scores > -5. )
copy_data = params[above_avg]
copy_scores = scores[above_avg]

# find unique
b = np.ascontiguousarray(copy_data).view(np.dtype((np.void, copy_data.dtype.itemsize * copy_data.shape[1])))
_, idx = np.unique(b, return_index=True)

# remove all repeats
copy_data = copy_data[idx]
copy_scores = copy_scores[idx]
print np.shape(copy_scores)
print "shape of unique data = ",len(copy_data)
def scatter(x,name,colors):

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    cNorm  = plt.matplotlib.colors.Normalize(vmin=np.min(colors), vmax=np.max(colors))
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,c=colors,cmap=cm,norm=cNorm)
    plt.colorbar(sc)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.savefig('parameter_clustering'+ name, dpi=150)
    plt.show()
    return


x = TSNE(metric='euclidean').fit_transform(copy_data)
x = StandardScaler().fit_transform(x)
scatter(x,'tsne_earm_traces_4_chains_100000_samples_0point1_std',copy_scores,)

X_2d = bh_sne(copy_data)
scatter(X_2d,'tsne_earm_traces_4_chains_100000_samples_0point1_std_bh',copy_scores)

#x = TSNE(metric='euclidean').fit_transform(params2.T)
#x = StandardScaler().fit_transform(x)
#scatter(x,'tmp',colors2[:len(params2.T)])

#X_2d = bh_sne(params2.T)
#scatter(X_2d,'pso_tsne',colors2[:len(params2.T)])