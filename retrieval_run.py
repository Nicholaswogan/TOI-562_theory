"""
Author: Natasha E. Batalha
Email: natasha.e.batalha@nasa.gov
"""
import os
import numpy as np
import pickle
import picaso.justdoit as jdi
import ultranest
import ultranest.stepsampler
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)
import warnings
warnings.filterwarnings('ignore')

"""
This run script runs all the models that we have setup in setup.py: 

The basic workflow in this script is: 
1) Load your data and store it in a convenient dictionary 
2) Define what data you want to test 
3) Define your likelihood function 
4) Specify what model set you are interested in testing (you can run these in a loop if they are very fast or one at a time)
5) Setup a directory output structure (here I create a new directory for each data provider, and model type)
6) Run Ultranest 
"""

# command line if you want to run this in parallel
# mpiexec -n numprocs python -m mpi4py pyfile
# for example: mpiexec -n 5 python -m mpi4py run_pymulti.py


# Here is your main file with all priors, and model sets
from retrieval_setup import param_set, model_set, prior_set
import utils

#2) Define what data you want to test 
#for running purposes, I just need to change data tag, and POC below. 
tag = 'TOI562-01'
data_tag = '30pix'
POC = 'JA'
data_file = 'data/TOI562_30pix_JA.csv'
DATA_DICT = utils.get_data_dict(data_file)

#3) Define your likelihood function 
def loglike(cube):
    data = DATA_DICT
    y, e = data['rprs2'], data['rprs2_err']

    # compute model spectra and normalize so we dont have to change our priors on baseline for every new planet
    resulty = MODEL(cube, data)
    if not PHYSICAL_MODEL:
        resulty = resulty*np.std(y) + np.mean(y)
    
    if 'logf' in LIST_PARAMS: 
        idx = LIST_PARAMS.index('logf')
        logf = cube[idx]
        sigma2 = e**2 + (10**(logf))**2
        loglikelihood = -0.5*np.sum((y-resulty)**2/sigma2 + np.log(2*np.pi*sigma2))
    else: 
        loglikelihood=-0.5*np.sum((y-resulty)**2/e**2)
    return loglikelihood

#4) Specify what model set you are interested in testing (you can run these in a loop if they are very fast or one at a time)

models_types = ['mh_cld_logf']
PHYSICAL_MODEL = True # for unphysical models we normalize the data to compute the log likelihood 

#5) Setup a directory output structure (here I create a new directory for each data provider, and model type)
out_dir = f'ultranest/{POC}/{data_tag}'
if not os.path.isdir(out_dir): 
    os.mkdir(out_dir) 
    
#6) Ultranest kwargs (fairly standard you should not need to change these)
multi_kwargs = {'resume':True,#'resume-similar',
                'warmstart_max_tau':-1,#0.7, #only used for resume-similar (small changes in likelihood. 0=very conservative, 1=very negligent) 
                'n_live_points':'200*nparam',
                'max_ncalls':None,
                'dlogz': 0.01}
       
##################### REST SHOULD NOT NEED TO BE EDITED #####################
for ir, retrieval_type in enumerate(models_types):
    
    print(retrieval_type)
    #get what parameters we want
    params = getattr(param_set, retrieval_type);Nparam=len(params.split(','))
    LIST_PARAMS = params.split(',')
    #get the model we want to test 
    MODEL = getattr(model_set, retrieval_type)
    #and the associated prior
    PRIOR = getattr(prior_set, retrieval_type)
    
    #now that we know the nubmer of parameters we can define the live points for the sampler
    if isinstance(multi_kwargs['n_live_points'],str):
        multi_kwargs['n_live_points']= (Nparam)*int(multi_kwargs['n_live_points'].split('*')[0])

    #lets create a nice json that gives us meta data about our run. 
    jdi.json.dump({
        'tag':tag, 
        'data_file':data_file, 
        'retrieval_type': retrieval_type,
        'nparams': Nparam, 
        'params':params, 
        'n_live_points': multi_kwargs['n_live_points'], 
        'max_ncalls':multi_kwargs['max_ncalls'],
        'dlogz':multi_kwargs['dlogz']
    } , open(f'{out_dir}/{tag}_{retrieval_type}.json','w'))
    
    #create a different directory for each run
    path = os.path.join(out_dir, retrieval_type)
    if not os.path.isdir(path): 
        os.mkdir(path)
    print('PATH',path)

    # Finally, run ultranest and save output! 
    sampler = ultranest.ReactiveNestedSampler(
                    params.split(','),
                    loglike,
                    PRIOR,
                    log_dir=path, resume=multi_kwargs['resume'], 
                    warmstart_max_tau=multi_kwargs['warmstart_max_tau'])
    
    results = sampler.run(min_num_live_points=multi_kwargs['n_live_points'], max_ncalls=multi_kwargs['max_ncalls'], dlogz=multi_kwargs['dlogz'])
    
    #dump results of pick too in case you want it for later (ultranest though will save what you need)
    pickle.dump(results, open(f'{out_dir}/{tag}_{retrieval_type}.pk','wb'))
   