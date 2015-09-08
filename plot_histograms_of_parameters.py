import pickle
from earm.lopez_embedded import model 
import pylab as plt
import numpy as np
import scipy.interpolate
cm = plt.cm.get_cmap('RdBu')


traces = pickle.load(open('test_traces2.p'))

params = [ i for i in traces['params']]
params = np.asarray(params)
params = params.reshape(np.shape(params)[0]*np.shape(params)[1],np.shape(params)[2])
#params = params[:100,:]
k_names = [p.name for p in model.parameters_rules()]
#fig = plt.figure(figsize=(10,10))
#for i in range(25):
#    ax =fig.add_subplot(5,5,i)
#    weights = np.ones_like(params[:,i])/float(len(params[:,i]))
#    ax.hist(params[:,i],100,weights=weights)
    #plt.title(k_names[i])
#    ax.set_xticklabels([])
#    ax.set_yticklabels([])
#fig.tight_layout()
#plt.show()
mat = np.zeros((len(model.parameters_rules()),len(model.parameters_rules())))
for i in range(105):
    for j in range(i+1,105):
        plt.plot(params[:,i],params[:,j],'.')
        plt.savefig('p%s_vs_p%s.png'%(i,j))
        plt.close()
        cov = np.corrcoef(np.vstack((params[:,i],params[:,j])))
        mat[i,j] = cov[0,1]
plt.imshow(mat+mat.T,interpolation='nearest',cmap=cm,
           vmin=-1,vmax=1)
plt.colorbar()
plt.show()