# -*- coding: utf-8 -*-


import numpy as np
import pysb.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import earm
from earm.lopez_embedded import model as EARM
from nightmare import Nightmare
import pickle


obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']
# Total starting amounts of proteins in obs_names, for normalizing simulations
obs_totals = [EARM.parameters['Bid_0'].value,
              EARM.parameters['PARP_0'].value]

earm_path = os.path.dirname(earm.__file__).rstrip('earm')

data_path = os.path.join(earm_path, 'xpdata', 'forfits',
                         'EC-RP_IMS-RP_IC-RP_data_for_models.csv')
exp_data = np.genfromtxt(data_path, delimiter=',', names=True)

# Model observable corresponding to the IMS-RP reporter (MOMP timing)
momp_obs = 'aSmac'
# Mean and variance of Td (delay time) and Ts (switching time) of MOMP, and
# yfinal (the last value of the IMS-RP trajectory)
momp_obs_total = EARM.parameters['Smac_0'].value
momp_data = np.array([9810.0, 180.0, momp_obs_total])
momp_var = np.array([7245000.0, 3600.0, 1e4])

ntimes = len(exp_data['Time'])
tmul = 10
tspan = np.linspace(exp_data['Time'][0], exp_data['Time'][-1],
                    (ntimes-1) * tmul + 1)

rate_params = EARM.parameters_rules()
param_values = np.array([p.value for p in EARM.parameters])
rate_mask = np.array([p in rate_params for p in EARM.parameters])
k_ids = [p.value for p in EARM.parameters_rules()]
nominal_values = np.array([p.value for p in EARM.parameters])
xnominal = np.log10(nominal_values[rate_mask])
bounds_radius = 2
solver = pysb.integrate.Solver(EARM, tspan, integrator='vode', rtol=1e-6, atol=1e-6,)

def display(position):

    exp_obs_norm = exp_data[data_names].view(float).reshape(len(exp_data), -1).T
    var_norm = exp_data[var_names].view(float).reshape(len(exp_data), -1).T
    std_norm = var_norm ** 0.5
    Y=np.copy(position)
    param_values[rate_mask] = 10 ** Y
    solver.run(param_values)
    obs_names_disp = obs_names + ['aSmac']
    sim_obs = solver.yobs[obs_names_disp].view(float).reshape(len(solver.yobs), -1)
    totals = obs_totals + [momp_obs_total]
    sim_obs_norm = (sim_obs / totals).T
    colors = ('r', 'b')
    for exp, exp_err, sim, c in zip(exp_obs_norm, std_norm, sim_obs_norm, colors):
        plt.plot(exp_data['Time'], exp, color=c, marker='.', linestyle=':')
        plt.errorbar(exp_data['Time'], exp, yerr=exp_err, ecolor=c,
                     elinewidth=0.5, capsize=0)
        plt.plot(solver.tspan, sim, color=c)
    plt.plot(solver.tspan, sim_obs_norm[2], color='g')
    plt.vlines(momp_data[0], -0.05, 1.05, color='g', linestyle=':')
    plt.show()


def likelihood(position):
    Y=np.copy(position)
    param_values[rate_mask] = 10 ** Y
    #changes={}
    #changes['Bid_0'] = 0
    #solver.run(param_values)
    #ysim_momp = solver.yobs[momp_obs]
    #if np.nanmax(ysim_momp) == 0:
    #    ysim_momp_norm = ysim_momp
    #else:
    #    return 100000,
        #return (100000,100000,100000)
    solver.run(param_values)
    for obs_name, data_name, var_name, obs_total in \
            zip(obs_names, data_names, var_names, obs_totals):
        ysim = solver.yobs[obs_name][::tmul]
        ysim_norm = ysim / obs_total
        ydata = exp_data[data_name]
        yvar = exp_data[var_name]
        if obs_name == 'mBid':
            e1 = np.sum(np.exp(-1.*(ydata - ysim_norm) ** 2 / (2 * yvar))) 
        else:
            e2 = np.sum(np.exp(-1.*(ydata - ysim_norm) ** 2 / (2 * yvar))) 
    ysim_momp = solver.yobs[momp_obs]
    if np.nanmax(ysim_momp) == 0:
        print 'No aSmac!'
        ysim_momp_norm = ysim_momp
        t10 = 0
        t90 = 0
    else:
        ysim_momp_norm = ysim_momp / np.nanmax(ysim_momp)
        st, sc, sk = scipy.interpolate.splrep(tspan, ysim_momp_norm)
        try:
            t10 = scipy.interpolate.sproot((st, sc-0.10, sk))[0]
            t90 = scipy.interpolate.sproot((st, sc-0.90, sk))[0]
        except IndexError:
            t10 = 0
            t90 = 0
    td = (t10 + t90) / 2
    ts = t90 - t10
    yfinal = ysim_momp[-1]
    momp_sim = [td, ts, yfinal]
    e3 = np.sum(np.exp(-1*(momp_data - momp_sim) ** 2 / (2 * momp_var)))
    error = -1.*np.log(e1) + -1.*np.log(e2) + -1.*np.log(e3)
    #print error
    return -1*error,
    #return (e1, e2, e3,)

if "__main__" == __name__:
    
    nm = Nightmare(EARM,likelihood,xnominal,'test')
    nm.run_pso(8, 25,200)
    ranked = nm.pso.return_ranked_populations()
    np.save('ndim_banana_seed.npy',ranked)
    # for i in nm.pso_results:
    #     print i['params']
    #     display(i['params'])
    traces = nm.run_DREAM(nsamples=500000)
    from pymc.backends import text
    text.dump('earm_traces_from_pymc_save_500000', traces)    
        
    dictionary_to_pickle = {}
    
    for dictionary in traces:
        for var in dictionary:
            dictionary_to_pickle[var] = traces[var] 
    
    pickle.dump(dictionary_to_pickle, open('test_traces_500000.p', 'wb'))
    # for n,each in enumerate(dictionary_to_pickle['params'][0]):
    #     plt.plot(n, likelihood(each),'or')
    # plt.plot(traces['error'][0])
    # 
    # plt.show()
    
    quit()


