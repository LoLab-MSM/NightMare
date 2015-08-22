#Clustering of parameter vectors

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cm = plt.cm.get_cmap('RdYlBu')


data = pickle.load(sys.argv[1])
print "shape of all data = ",len(data)
params = [ i for i in data['params']]
params = np.asarray(params)
params = params.reshape(np.shape(params)[0]*np.shape(params)[1],np.shape(params)[2])

scores = [ i for i in data['error']]
scores = np.asarray(scores)
scores = scores.reshape(np.shape(scores)[0]*np.shape(scores)[1])


above_avg= np.where( scores > 0. )
copy_data = params[above_avg]
copy_scores = scores[above_avg]

b = np.ascontiguousarray(copy_data).view(np.dtype((np.void, copy_data.dtype.itemsize * copy_data.shape[1])))

_, idx = np.unique(b, return_index=True)

copy_data = copy_data[idx]
copy_scores = copy_scores[idx]



def scatter(x, colors, name):

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    cNorm  = plt.matplotlib.colors.Normalize(vmin=np.min(colors), vmax=np.max(colors))
    print np.shape(x[:,0])
    print np.shape(colors)
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,c=colors,cmap=cm,norm=cNorm)
    plt.colorbar(sc)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.savefig('parameter_clustering'+ name, dpi=150)
    plt.close(f)
    return


x = TSNE(metric='euclidean',random_state=20150101).fit_transform(copy_data)
x = StandardScaler().fit_transform(x)

scatter(x,copy_scores,'parameters1')


