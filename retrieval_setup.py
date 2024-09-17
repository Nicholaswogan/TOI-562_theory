"""
Author: Natasha E. Batalha
Email: natasha.e.batalha@nasa.gov
"""
import numpy as np
import picaso.justdoit as jdi

import planets
import utils

"""
This setup script setups up one function and four classes: 

Function
--------
get_data : 
    - This is a top level function you can create to read in the data you want to test 
    - You can look through it and change as necessary. As you can see everyone has fun different file structures. I usually add e.g. "Nicole" or "NW" to a file if the user has not put their name in their filename. 

Classes
-------
param_set: 
    - defines each of the models you want to run and their associated free parameters 
    - we keep track of these through a string list separated by commas. For example for the model y=mx+b your param_set would be line='m,b'
guesses_set: 
    - for each of the models you defined in param_set we now want to have a set of guesses on hand that can help us test our models 
    - we keep track of these through a list of numerical values. For example, in our example of y=mx+b your guesse_set might be line= [-1,1]
model_set: 
    - for each model name, we now must create a function for each of them. 
    - this is quite literally the model e.g. "def line(cube,x): m,b=cube;return m*x+b
prior_set: 
    - and lastly, what are the bounds of your prior for each parameter you setup.
"""

def create_picaso():
    opas = jdi.opannection(wave_range=[2.7,5.2])
    plan_og = jdi.inputs()
    plan_og.phase_angle(0)
    plan_og.gravity(mass=planets.TOI562_01.mass, mass_unit=jdi.u.Unit('M_earth'),
                    radius=planets.TOI562_01.radius, radius_unit=jdi.u.Unit('R_earth'))
    plan_og.star(opas, planets.TOI562.Teff, planets.TOI562.metal, planets.TOI562.logg, 
              radius=planets.TOI562.radius, radius_unit = jdi.u.Unit('R_sun'),database='phoenix')
    return opas, plan_og

# Global variables
PICASO_OPAS, PICASO_PLAN = create_picaso()
METALLICITY_CALCULATOR = utils.MetalicityCalculator()
PLANET_TEQ = planets.TOI562_01.Teq

def log10x_to_mix(log10x):
    x = 10.0**log10x
    mix = x/np.sum(x)
    return mix

class param_set: 
    """
    This sets up what your free parameters are in each model you want to test
    """
    step_line = 'b1,b2'
    step_line_logf = 'b1,b2,logf'
    gauss_free_step_line = 'lam0,sig,Amp,cst_nrs1,cst_nrs2'
    gauss_free_step_line_logf = 'lam0,sig,Amp,cst_nrs1,cst_nrs2,logf'
    mh_cld='logmh,cldp,cst_nrs1,cst_nrs2'
    mh_cld_logf='logmh,cldp,cst_nrs1,cst_nrs2,logf'
    h2oh2_cld='log10xh2,log10xh2o,cldp,cst_nrs1,cst_nrs2'
    h2oh2_cld_logf='log10xh2,log10xh2o,cldp,cst_nrs1,cst_nrs2,logf'

class model_set: 
    """
    Here are the models that should be associated with each of your parameters and guesses. 
    """
        
    def step_line(cube, data):
        wlgrid = data['wv']
        spec = wlgrid*0
        spec[wlgrid<3.78] = spec[wlgrid<3.78] + cube[0]
        spec[wlgrid>=3.78] = spec[wlgrid>=3.78] + cube[1]
        return spec
    
    step_line_logf = step_line
    
    def gauss_free_step_line(cube, data):
        wlgrid = data['wv']
        lam0, logsig, logAmp, cst_nr1, cst_nr2 = cube[0], cube[1], cube[2], cube[3], cube[4]
        sig = 10**logsig
        Amp = 10**logAmp
        val = np.empty(wlgrid.shape[0])
        val[wlgrid<3.78] = (Amp*np.exp(-(wlgrid[wlgrid<3.78]-lam0)**2/sig**2)+cst_nr1)
        val[wlgrid>=3.78] = (Amp*np.exp(-(wlgrid[wlgrid>=3.78]-lam0)**2/sig**2)+cst_nr2)
        return val
    
    gauss_free_step_line_logf = gauss_free_step_line
    
    def mh_cld(cube, data): 
        log10MH, log_cld_top, cst_nr1, cst_nr2 = cube[0], cube[1], cube[2], cube[3]

        # PICASO and chemisty
        opas = PICASO_OPAS
        plan = PICASO_PLAN
        mc = METALLICITY_CALCULATOR
        
        # Compute a temperature profile
        pt = plan.guillot_pt(PLANET_TEQ, T_int=0.0)
        T = pt['temperature'].to_numpy()
        P = pt['pressure'].to_numpy()

        # Compute chemical equilibrium along profile
        CtoO = 1.0 # C/O ratio 1x solar (same as solar)
        df = mc.solve_picaso(T, P, CtoO, log10MH, rainout_condensed_atoms=False)
        df['temperature'] = T
        df['pressure'] = P
        plan.atmosphere(df=df)

        # Add a cloud
        cloud_bottom = np.log10(P[-1])
        log_dp = cloud_bottom - log_cld_top
        plan.clouds(g0=[0.9], w0=[0.9], opd=[10], p=[cloud_bottom], dp=[log_dp])

        # Run picaso
        df_picaso = plan.spectrum(opas, calculation='transmission', full_output=True)
        x, y = df_picaso['wavenumber'], df_picaso['transit_depth']
        wno, model = jdi.mean_regrid(df_picaso['wavenumber'], df_picaso['transit_depth'] , R=1000)
        wv_model = 1e4/wno[::-1].copy()
        rprs2_model = model[::-1].copy()
        
        # Rebin picaso to data
        _, biny = utils.rebin_picaso_to_data(wv_model, rprs2_model, data['wv'], data['wv_bin_hw'])

        # Add offsets
        biny[data['wv']<3.78] += cst_nr1
        biny[data['wv']>=3.78] += cst_nr2
        
        return biny
    
    mh_cld_logf = mh_cld

    def h2oh2_cld(cube, data): 
        log10xH2, log10xH2O, log_cld_top, cst_nr1, cst_nr2 = cube[0], cube[1], cube[2], cube[3], cube[4]

        # PICASO and chemisty
        opas = PICASO_OPAS
        plan = PICASO_PLAN

        # Compute mixing ratios from input x values
        mix = log10x_to_mix(np.array([log10xH2, log10xH2O]))
        f_H2 = mix[0]
        f_H2O = mix[1]

        # Compute a temperature profile
        df = plan.guillot_pt(PLANET_TEQ, T_int=0.0)
        P = df['pressure'].to_numpy()

        # Add chemistry
        df['H2'] = np.ones(P.shape[0])*f_H2
        df['H2O'] = np.ones(P.shape[0])*f_H2O
        plan.atmosphere(df=df)

        # Add a cloud
        cloud_bottom = np.log10(P[-1])
        log_dp = cloud_bottom - log_cld_top
        plan.clouds(g0=[0.9], w0=[0.9], opd=[10], p=[cloud_bottom], dp=[log_dp])

        # Run picaso
        df_picaso = plan.spectrum(opas, calculation='transmission', full_output=True)
        x, y = df_picaso['wavenumber'], df_picaso['transit_depth']
        wno, model = jdi.mean_regrid(df_picaso['wavenumber'], df_picaso['transit_depth'] , R=1000)
        wv_model = 1e4/wno[::-1].copy()
        rprs2_model = model[::-1].copy()

        # Rebin picaso to data
        _, biny = utils.rebin_picaso_to_data(wv_model, rprs2_model, data['wv'], data['wv_bin_hw'])

        # Add offsets
        biny[data['wv']<3.78] += cst_nr1
        biny[data['wv']>=3.78] += cst_nr2
        
        return biny
    
    h2oh2_cld_logf = h2oh2_cld

class prior_set: 
    """
    And, for each model we need a prior bound for each of the gree parameters
    """
    def step_line(cube):
        params = cube.copy()
        minv = -5 #.0001
        maxv = 5 #.005
        params[0] = minv + (maxv-minv)*params[0]
        minv = -5 #.0001
        maxv = 5 #.005
        params[1] = minv + (maxv-minv)*params[1]
        return params  

    def step_line_logf(cube):
        params = cube.copy()
        minv = -5 #.0001
        maxv = 5 #.005
        params[0] = minv + (maxv-minv)*params[0]
        minv = -5 #.0001
        maxv = 5 #.005
        params[1] = minv + (maxv-minv)*params[1]

        # logf
        minn = -7
        maxx = -3
        params[2] =  minn + (maxx-minn)*params[2]   
        return params  
    
    def gauss_free_step_line(cube):  
        params = cube.copy()
        lam0, logsig, logAmp, cst_nrs1, cst_nrs2 = params
        mina =-2
        maxa =1.5 
        logAmp=mina+(maxa-mina)*logAmp
        min_wavelength=3
        max_wavelength=5.2
        lam0=min_wavelength+(max_wavelength-min_wavelength)*lam0 
        min_width = np.log10(0.01)
        max_width = np.log10(2)
        logsig=min_width+(max_width-min_width)*logsig
        minv = -5
        maxv = 5
        cst_nrs1 = minv + (maxv-minv)*cst_nrs1
        minv = -5
        maxv = 5
        cst_nrs2 = minv + (maxv-minv)*cst_nrs2
        params =[lam0, logsig, logAmp,cst_nrs1, cst_nrs2]
        return params 
    
    def gauss_free_step_line_logf(cube):  
        params = cube.copy()
        lam0, logsig, logAmp, cst_nrs1, cst_nrs2, logf = params
        mina =-2
        maxa =1.5 
        logAmp=mina+(maxa-mina)*logAmp
        min_wavelength=3
        max_wavelength=5.2
        lam0=min_wavelength+(max_wavelength-min_wavelength)*lam0 
        min_width = np.log10(0.01)
        max_width = np.log10(2)
        logsig=min_width+(max_width-min_width)*logsig
        minv = -5
        maxv = 5
        cst_nrs1 = minv + (maxv-minv)*cst_nrs1
        minv = -5
        maxv = 5
        cst_nrs2 = minv + (maxv-minv)*cst_nrs2

        # logf
        minn = -7
        maxx = -3
        logf =  minn + (maxx-minn)*logf 

        params = [lam0, logsig, logAmp,cst_nrs1, cst_nrs2, logf]
        return params 

    def mh_cld(cube):
        params = cube.copy()
        min_mh = 0
        max_mh = 3
        params[0] = min_mh + (max_mh-min_mh)*params[0]

        min_cldtop = -5
        max_cldtop = 1.0
        params[1] = min_cldtop + (max_cldtop-min_cldtop)*params[1]

        min_offset = -1000e-6
        max_offset = +1000e-6
        params[2] = min_offset + (max_offset-min_offset)*params[2]
        params[3] = min_offset + (max_offset-min_offset)*params[3]
        return params
        
    def mh_cld_logf(cube):
        params = cube.copy()
        min_mh = 0
        max_mh = 3
        params[0] = min_mh + (max_mh-min_mh)*params[0]

        min_cldtop = -5
        max_cldtop = 1.0
        params[1] = min_cldtop + (max_cldtop-min_cldtop)*params[1]

        min_offset = -1000e-6
        max_offset = +1000e-6
        params[2] = min_offset + (max_offset-min_offset)*params[2]
        params[3] = min_offset + (max_offset-min_offset)*params[3]

        # logf
        minn = -7
        maxx = -3
        params[4] =  minn + (maxx-minn)*params[4]        
        return params
    
    def h2oh2_cld(cube):
        params = cube.copy()
        min_log10x = -5
        max_log10x = 0

        # H2
        params[0] = min_log10x + (max_log10x-min_log10x)*params[0]

        # H2O
        params[1] = min_log10x + (max_log10x-min_log10x)*params[1]

        min_cldtop = -5
        max_cldtop = 1.0
        params[2] = min_cldtop + (max_cldtop-min_cldtop)*params[2]

        min_offset = -1200e-6
        max_offset = +1200e-6
        params[3] = min_offset + (max_offset-min_offset)*params[3]
        params[4] = min_offset + (max_offset-min_offset)*params[4]
        return params
    
    def h2oh2_cld_logf(cube):
        params = cube.copy()
        min_log10x = -5
        max_log10x = 0

        # H2
        params[0] = min_log10x + (max_log10x-min_log10x)*params[0]

        # H2O
        params[1] = min_log10x + (max_log10x-min_log10x)*params[1]

        min_cldtop = -5
        max_cldtop = 1.0
        params[2] = min_cldtop + (max_cldtop-min_cldtop)*params[2]

        min_offset = -1200e-6
        max_offset = +1200e-6
        params[3] = min_offset + (max_offset-min_offset)*params[3]
        params[4] = min_offset + (max_offset-min_offset)*params[4]

        # logf
        minn = -7
        maxx = -3
        params[5] =  minn + (maxx-minn)*params[5]
        return params
