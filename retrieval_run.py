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

#command line if you want to run this in parallel
#mpiexec -n numprocs python -m mpi4py pyfile
#for example: mpiexec -n 5 python -m mpi4py run_pymulti.py


#Here is your main file with all priors, and model sets
from setup import *


#1) Load data which returns wavelenght, transit spectra, transit error (rp/rs)^2
#NW 
tag = 'TOI776-01' #helpful for metadata and filenaming

dirs = 'data/NW/v2'
v1 = os.path.join(dirs,
        'TOI77601_visit1_spectrum.csv')
v2 = os.path.join(dirs,
        'TOI77601_visit2_spectrum.csv')

j = os.path.join(dirs,
        'JWST_77601_joint_incPrior_e.csv')

fname_noeng = os.path.join(dirs,
        'TOI836_30pixel_Nicole.csv')
NW = {'v1':v1, 'v2':v2, 'j':j}

#JK
dirs = 'data/JK/Feb2024/'
v1 = os.path.join(dirs,
        'Tiberius-TOI-776p01-spectrum_n1v1.txt')
v2 = os.path.join(dirs,
        'Tiberius-TOI-776p01-spectrum_n1v2.txt')
j = os.path.join(dirs,
        'Tiberius_combined_spectrum_Feb2024.txt')

JK ={'v1':v1, 'v2':v2, 'j':j}#for the sanity of the rest of the code, dont change v1, v2, j

#now I have a convenient dictionary that contains all the data I want to eventually test 
all_data = {'NW':NW, 'JK':JK}

#2) Define what data you want to test 
#for running purposes, I just need to change data tag, and POC below. 
data_tag = 'j'
POC = 'JK'
data_file = all_data[POC][data_tag]
x,y,e = get_data(data_file,rebin=False)


#3) Define your likelihood function 
def loglike(cube):
    #compute model spectra and normalize so we dont have to change our priors on baseline for every new planet
    resulty = MODEL(cube,x)
    if not physical_model :
        resulty=resulty*np.std(y)+np.mean(y)
    
    #logf
    if 'logf' in list_params: 
        idx = list_params.index('logf')
        logf=cube[-1]
        sigma2 = e**2 + (10**(logf))**2
        loglikelihood = -0.5*np.sum((y-resulty)**2/sigma2 + np.log(2*np.pi*sigma2))
    else: 
        loglikelihood=-0.5*np.sum((y-resulty)**2/e**2)
    return loglikelihood

#    sigma2 = edata**2 + (10**(logf))**2
#    loglikelihood = -0.5*np.sum((ydata-y_model_all)**2/sigma2 + np.log(2*np.pi*sigma2))


#4) Specify what model set you are interested in testing (you can run these in a loop if they are very fast or one at a time)

#models_types = ['slope_0_line','step_line','line','gauss_free','gauss_free_nrs1','gauss_free_nrs2','gauss_ch4','gauss_line_free']#
models_types = ['mh_cld_logf']#['ch4h2_cld','h2oh2_cld','co2h2_cld']
physical_model = True #for unphysical models we normalize the data to compute the log likelihood 

#5) Setup a directory output structure (here I create a new directory for each data provider, and model type)
out_dir = f'ultranest/{POC}/{data_tag}'
if not os.path.isdir(out_dir): 
    os.mkdir(out_dir) 
    
#6) Ultranest kwargs (fairly standard you should not need to change these)
multi_kwargs = {'resume':True,#'resume-similar',
                'warmstart_max_tau':-1,#0.7, #only used for resume-similar (small changes in likelihood. 0=very conservative, 1=very negligent) 
                'n_live_points':'50*nparam',
                'max_ncalls':None}
    
    
##################### REST SHOULD NOT NEED TO BE EDITED #####################
for ir, retrieval_type in enumerate(models_types):
    
    print(retrieval_type)
    #get what parameters we want
    params = getattr(param_set, retrieval_type);Nparam=len(params.split(','))
    list_params = params.split(',')
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
        'max_ncalls':multi_kwargs['max_ncalls']
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
    
    results = sampler.run(min_num_live_points=multi_kwargs['n_live_points'], max_ncalls=multi_kwargs['max_ncalls'])
    
    #dump results of pick too in case you want it for later (ultranest though will save what you need)
    pickle.dump(results, open(f'{out_dir}/{tag}_{retrieval_type}.pk','wb'))
   