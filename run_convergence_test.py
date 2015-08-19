import pickle
from helper_fxns import convert_param_vec_dict_to_param_dict
from helper_fxns import print_convergence_summary
from earm.lopez_embedded import model 
import pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import PCA as mlabPCA
import scipy.interpolate
from pysb.tools.stochkit import StochKitSimulator
from pysb.tools.bngSSA import BNGSSASimulator
from run_nightname import likelihood as like
from run_nightname import display
cm = plt.cm.get_cmap('RdYlBu')


traces = pickle.load(open('test_traces_500000.p'))

scores = [ i for i in traces['error']]
scores = np.asarray(scores)
scores = scores.reshape(np.shape(scores)[0]*np.shape(scores)[1])
params = [ i for i in traces['params']]

#np.savetxt('trace1.txt',traces['params'][0])
#np.savetxt('trace2.txt',traces['params'][1])
#np.savetxt('trace3.txt',traces['params'][2])
#np.savetxt('trace4.txt',traces['params'][3])
#np.savetxt('trace5.txt',traces['params'][4])
#np.savetxt('trace6.txt',traces['params'][5])
#np.savetxt('trace7.txt',traces['params'][6])
#np.savetxt('trace8.txt',traces['params'][7])
#par1 = np.loadtxt('trace1.txt')
# par2 = np.loadtxt('trace2.txt')
# par3 = np.loadtxt('trace3.txt')
# par4 = np.loadtxt('trace4.txt')
# par5 = np.loadtxt('trace5.txt')
# par6 = np.loadtxt('trace6.txt')
# par7 = np.loadtxt('trace7.txt')
# par8 = np.loadtxt('trace8.txt')


# def determine_unique(i):
#     data = np.loadtxt('trace%s.txt'%str(i))
#     b = np.ascontiguousarray(data).view(np.dtype((np.void, data.dtype.itemsize * data.shape[1])))
#     _, idx = np.unique(b, return_index=True)
#     return data[idx]
# data = []
# for i in range(1,8):
#     data.append(determine_unique(i))
# params = np.asarray(data)
# #params = np.column_stack((par1,par2,par3,par4,par5,par5,par7,par8))
# params = determine_unique(params)

params = np.asarray(params)
params = params.reshape(np.shape(params)[0]*np.shape(params)[1],np.shape(params)[2])

def gelman_rubin_trace_dict(params):
    
    means,Vars = [],[]
    n = len(params)
    for i in range(len(params)):
        means.append( np.mean(params[i],axis=0))
    vari = np.var(np.asarray(means),axis=0,ddof=1)
    for i in range(len(params)):
        Vars.append( np.var(params[i],axis=0,ddof=1))
    Mean = np.mean(np.asarray(Vars) ,axis=0)     
    Vhat = Mean*(n - 1)/n + vari/n
    return  Vhat

param_vec_dict = convert_param_vec_dict_to_param_dict(traces, model.parameters_rules())
#print_convergence_summary(param_vec_dict)

data = params
above_avg= np.where( scores > 0. )
copy_data = data[above_avg]
print len(copy_data)
copy_scores = scores[above_avg]


b = np.ascontiguousarray(copy_data).view(np.dtype((np.void, copy_data.dtype.itemsize * copy_data.shape[1])))

_, idx = np.unique(b, return_index=True)

copy_data = copy_data[idx]
copy_scores = copy_scores[idx]
print len(copy_data)
print len(copy_scores)
def plot_pca():
    pca = mlabPCA(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cNorm  = plt.matplotlib.colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))
    sc = ax.scatter(pca.Y[:,0],pca.Y[:,1],pca.Y[:,2],c=scores,cmap=cm,norm=cNorm)
    plt.colorbar(sc)
    plt.savefig('PCA.png',dpi=300)
    plt.show()


def plot_pca_top(data,scores,savename='PCA'):
    pca = mlabPCA(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cNorm  = plt.matplotlib.colors.Normalize(vmin=np.min(scores), vmax=np.max(scores))
    sc = ax.scatter(pca.Y[:,0],pca.Y[:,1],pca.Y[:,2],c=scores,cmap=cm,norm=cNorm)
    plt.colorbar(sc)
    plt.savefig('%s.png' % savename,dpi=300)
    plt.show()


plot_pca_top(copy_data,copy_scores,'rerun')
quit()
def likelihood(ysim_momp):
    death = 1
    if np.nanmax(ysim_momp) == 0:
        ysim_momp_norm = ysim_momp
        t10 = 0
        t90 = 0
        death =0
    else:
        ysim_momp_norm = ysim_momp / np.nanmax(ysim_momp)
        st, sc, sk = scipy.interpolate.splrep(solver.tspan, ysim_momp_norm)
        try:
            t10 = scipy.interpolate.sproot((st, sc-0.10, sk))[0]
            t90 = scipy.interpolate.sproot((st, sc-0.90, sk))[0]
        except IndexError:
            t10 = 0
            t90 = 0
    td = (t10 + t90) / 2
    ts = t90 - t10
    return td/3600, ts , death

def calculate_death_stats(name, traj,tout):
    TOD =np.zeros(len(traj))
    TS = np.zeros(len(traj))
    DEATH = 0  
    for n in range(len(traj)):
        td,ts,death = likelihood(traj[name][n])
        TOD[n]= td
        TS[n]=ts
        DEATH+=death
    print "Time of death average",np.average(TOD)
    print "Time taken for relase average",np.average(TS)
    print DEATH," out of ",len(traj)
    return np.average(TOD),np.average(TS),DEATH/len(traj)

def plot_mean_min_max(name, title=None):
    x = np.array([tr[:][name] for tr in trajectories]).T
    if not title:
        title = name
    plt.figure(title)
    plt.plot(tout.T, x, '0.5', lw=2, alpha=0.25) # individual trajectories
    plt.plot(tout[0], x.mean(1), 'k--', lw=3, label="Mean")
    plt.plot(tout[0], x.min(1), 'b--', lw=3, label="Minimum")
    plt.plot(tout[0], x.max(1), 'r--', lw=3, label="Maximum")
    plt.legend(loc=0)
    plt.xlabel('Time')
    plt.ylabel('Population of %s' %name)
    plt.show()

tspan = np.linspace(0, 20000, 101)
rate_params = model.parameters_rules()
param_values = np.array([p.value for p in model.parameters])
rate_mask = np.array([p in rate_params for p in model.parameters])
fraction_death = np.zeros(len(copy_data))
for i in range(5,len(copy_data)):
    print "Cost function of position",like(copy_data[i])
    param_values[rate_mask] = 10 ** copy_data[i]
    #display(copy_data[i])
    solver = StochKitSimulator(model,tspan )
    solver.run(tspan, param_values= param_values, n_runs=10, )
    tout = solver.tout
    trajectories = solver.get_yfull()
    td, ts, frac = calculate_death_stats('aSmac',solver.yobs,tout)
    fraction_death[i] = frac
plot_pca_top(copy_data,fraction_death,'fraction_pca')

quit()
def view_plots():
    plot_mean_min_max('aSmac')
    plot_mean_min_max('cPARP')
    plot_mean_min_max('mBid')
view_plots()
calculate_death_stats('aSmac',solver.yobs,tout)

