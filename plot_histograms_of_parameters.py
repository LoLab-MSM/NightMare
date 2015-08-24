import pickle
from helper_fxns import convert_param_vec_dict_to_param_dict
from helper_fxns import print_convergence_summary
from earm.lopez_embedded import model 
import pylab as plt
import numpy as np
import scipy.interpolate
cm = plt.cm.get_cmap('RdYlBu')


traces = pickle.load(open('test_traces2.p'))

params = [ i for i in traces['params']]
params = np.asarray(params)
params = params.reshape(np.shape(params)[0]*np.shape(params)[1],np.shape(params)[2])

k_names = [p.name for p in model.parameters_rules()]
fig = plt.figure(figsize=(10,10))
for i in range(25):
    ax =fig.add_subplot(5,5,i)
    weights = np.ones_like(params[:,i])/float(len(params[:,i]))
    ax.hist(params[:,i],100,weights=weights)
    #plt.title(k_names[i])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
fig.tight_layout()
plt.show()