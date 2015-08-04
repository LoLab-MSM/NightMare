from refactored_pso import PSO
import theano
import theano.tensor as t
import numpy as np
import pymc
import pylab as plt


class Nightmare():
    def __init__(self,model,cost_function,time,start_parameters,savename,bound=1):
        self.model = model
        self.cost_function = cost_function
        self.start_parameters = start_parameters
        self.bound = bound
        self.set_PSO()
        self.savename = savename
        
    def set_PSO(self,):
        self.pso = PSO()
        self.pso.set_cost_function(self.cost_function)
        self.pso.update_w = True
        self.pso.set_start_position(self.start_parameters)
        self.pso.set_bounds(self.bound)
        self.pso.set_speed(-0.5,0.5)
        
    def run_pso(self,nchains,nparticles,niterations):
        self.nchains = nchains
        self.pso_results = []
        for _ in range(nchains):
            self.pso.run(nparticles,niterations)
            self.pso_results.append({'params':np.asarray(self.pso.best)})
            self.pso.set_start_position(self.start_parameters)
            self.pso.best = None
    

    
    def run_DREAM(self):
        model = pymc.Model()
        with model:
            params = pymc.Normal('params', mu=self.start_parameters, 
                                 sd=np.array([self.bound]*len(self.start_parameters)),
                                 shape=(len(self.start_parameters)))
              
            global cost_function
            cost_function = self.cost_function
            
            error = DREAM_cost(params)
            error = pymc.Deterministic('error', error)

            nseedchains = 10*len(self.model.parameters_rules())
            
            step = pymc.Dream(variables=[params],
                              nseedchains=nseedchains, 
                              blocked=True,
                              start_random=False,
                              save_history=True,
                              parallel=True,
                              adapt_crossover=False,
                              verbose=False,
                              model_name = self.savename)

            

            trace = pymc.sample(10000, step,
                                start=self.pso_results, 
                                njobs=self.nchains,
                                use_mpi=False,
                                progressbar=True) 
            pymc.traceplot(trace[1000:],vars=['params','error'])
            plt.show()
            return trace

@theano.compile.ops.as_op(itypes=[t.dvector], otypes=[t.dscalar])
def DREAM_cost(parameters):
    global cost_function
    like = cost_function(parameters)
    return -1.*like[0]
            