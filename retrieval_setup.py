"""
Author: Natasha E. Batalha
Email: natasha.e.batalha@nasa.gov
"""
import pandas as pd
import numpy as np
import spectres
import picaso.justdoit as jdi


#if running M/H fits 
from photochem import zahnle_earth
from photochem.utils import photochem2cantera
photochem2cantera(zahnle_earth,'zahnle_earth_ct.yaml')
from metallicity import MetallicityCalculator

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

# For cases where we are running physical retrievals 
params = pd.read_csv('system_params.txt', sep='\t')
opas = jdi.opannection(wave_range=[2.7,5.2])
iname = 'TOI-776.01'
row = pd.DataFrame(dict(st_teff=params['starTeff (K)'].loc[params['name']==iname],
        st_logg=params['starlog(g)'].loc[params['name']==iname],
        st_metfe=params['star[Fe/H]'].loc[params['name']==iname],
        st_rad=params['starRadius(Rsun)'].loc[params['name']==iname],
        pl_radj=params['Radius (Rearth)'].loc[params['name']==iname] * (jdi.c.R_earth/jdi.c.R_jup).value,
        pl_bmassj=params['Mass (Mearth)'].loc[params['name']==iname] * (jdi.c.M_earth/jdi.c.M_jup).value,
        pl_eqt=params[ 'Teq (K)'].loc[params['name']==iname]))
plan_og = jdi.load_planet(row, opas)


def get_data(fname,rebin=False, weighted=False, common_wavelength=None): 
    
    if rebin: 
        if not isinstance(common_wavelength, str):
            raise Exception('Common_wavelength file was not provided')
        df = pd.read_csv(common_wavelength,
                        skiprows=1, header=None,
                        names=['um',
                            'tdepth', 'etdepth'])
        xrebin = df['um'].values
  
    
    if (('Nicole' in fname) or ('NW' in fname)): 
        #Wavelength(microns),(Rp/R*)^2,Error
        if '.nc' in fname: 
            ds = xr.load_dataset(fname) #'TOI836.02/PlanetSpectra/NW/transit-spectrum-TOI_836_02-joint-G395H-30pix-MCMC-Nicole.nc')

            x,y,e = (ds['central_wavelength'].values, 
                     ds['transit_depth'].values,
                     ds['transit_depth_error'].values,
                    )
        else: 
            df = pd.read_csv(fname,
                            skiprows=1, header=None,
                            names=['um',
                                'tdepth', 'etdepth'])
            x,y,e = (df['um'].values, 
                     df['tdepth'].values,
                     df['etdepth'].values,
                    )
        
    elif 'adams' in fname.lower(): 
        #two files for nirs1 and nirs2
        check = pd.read_csv(fname,sep='\s+',
                        skiprows=1, header=None)
        if check.shape[1]==4: 
            df = pd.read_csv(fname,sep='\s+',
                            skiprows=1, header=None,
                            names=['um', 'um_width', 
                               'tdepth', 'etdepth'])
            df = df.dropna()
        else: 
            df = pd.read_csv(fname,sep='\s+',
                            skiprows=1, header=None,
                            names=['um', 'um_width', 
                               'tdepth', 'etdepth1','etdepth2'])
            df = df.dropna()
            df['etdepth'] = (df['etdepth1'] + df['etdepth2'] )/2
        
        
        df = df.loc[df['tdepth']>0]
        x1,y1,e1 = (df['um'].values, 
                 df['tdepth'].values,
                 df['etdepth'].values,
                )
        fname = fname.replace('nrs1','nrs2')
        
        if check.shape[1]==4: 
            df = pd.read_csv(fname,sep='\s+',
                            skiprows=1, header=None,
                            names=['um', 'um_width', 
                               'tdepth', 'etdepth'])
            df = df.dropna()
        else: 
            df = pd.read_csv(fname,sep='\s+',
                            skiprows=1, header=None,
                            names=['um', 'um_width', 
                               'tdepth', 'etdepth1','etdepth2'])
            df = df.dropna()
            df['etdepth'] = (df['etdepth1'] + df['etdepth2'] )/2
            
        df = df.loc[df['tdepth']>0]
        x2,y2,e2 = (df['um'].values, 
                 df['tdepth'].values,
                 df['etdepth'].values,
                )
        
        x,y,e = (np.concatenate((x1,x2)), 
                np.concatenate((y1,y2)), 
                np.concatenate((e1,e2)))
    elif 'LA' in fname: 
        ds = xr.load_dataset(fname)
        x,y,e = (ds.coords['central_wavelength'].values, 
                 ds.data_vars['transit_depth'].values,
                 ds.data_vars['transit_depth_error'].values,
                )
        if isinstance(weighted,str): 
            
            flux_visit1 = y 
            error_visit1 = e
            
            ds2 = xr.load_dataset(weighted)
            x, flux_visit2, error_visit2 = (ds2.coords['central_wavelength'].values, 
                 ds2.data_vars['transit_depth'].values,
                 ds2.data_vars['transit_depth_error'].values,
                )
            
            mean=((flux_visit1/error_visit1**2) +(flux_visit2/error_visit2**2) )/((1/(error_visit1**2))+(1/(error_visit2**2)))
            sigma=np.sqrt(1/((1/(error_visit1**2))+(1/(error_visit2**2))))
            
            y = mean
            e = sigma 
            
    elif 'scarsdale' in fname: 
        df = pd.read_csv(fname, header=None,
                        names=['um', 'tdepth', 'edepth'])
        x,y,e = (df['um'].values, 
                 df['tdepth'].values,
                 df['edepth'].values,
                )
        
    elif 'munazza' in fname: 
        df = pd.read_csv(fname, header=None,skiprows=1,sep='\s+',
                        names=['bins_center', 'bin_wdth', 
                               'transit_depth','transit_depth_err']).astype(float)
        x,y,e = (df['bins_center'].values, 
                 df['transit_depth'].values,
                 df['transit_depth_err'].values,
                )
    elif 'Tiberius' in fname: 
        df = pd.read_csv(fname, header=None,skiprows=1,sep='\s+',
                        names=['bins_center', 'bin_wdth', 
                               'transit_depth','transit_depth_err']).astype(float)
        x,y,e = (df['bins_center'].values, 
                 df['transit_depth'].values,
                 df['transit_depth_err'].values,
                )
        if 'n1' in fname:
            df = pd.read_csv(fname.replace('n1','n2'), header=None,skiprows=1,sep='\s+',
                            names=['bins_center', 'bin_wdth', 
                                   'transit_depth','transit_depth_err']).astype(float)
            x2,y2,e2 = (df['bins_center'].values, 
                     df['transit_depth'].values,
                     df['transit_depth_err'].values,
                    )
            x = np.concatenate((x,x2))
            y = np.concatenate((y,y2))
            e = np.concatenate((e,e2))

        if np.min(y)>1: 
            #convert from ppm units if y is greater than 1
            y = y/1e6
            e = e/1e6
        
    if rebin: 
        xrebin=xrebin[((xrebin<max(x)) & (xrebin>min(x)) )]
        y, e = spectres.spectres(xrebin, x[~np.isnan(y)]
                                 ,y[~np.isnan(y)],e[~np.isnan(y)])
        x = xrebin
        
    return x,y,e


class param_set: 
    """
    This sets up what your free parameters are in each model you want to test
    """
    slope_0_line = 'b' #a model to get the baseline (zero slope)
    line = 'm,b' #a model to test for a slope
    step_line = 'b1,b2' #a model to test for offsets
    gauss_free = 'lam0,sig,Amp,cst' #a model to test if there are any gaussian like features in your data set
    gauss_free_nrs1= 'lam0,sig,Amp,cst_nrs1,cst_nrs2' #same as above but let's add an offset and look only in nrs1 detector
    gauss_free_nrs2= 'lam0,sig,Amp,cst_nrs1,cst_nrs2' #same as above but let's do only nrs2
    gauss_ch4 = 'sig,Amp,cst_nrs1,cst_nrs2' #lets test specifically for something located where CH4 is
    gauss_line_free = 'lam0,sig,Amp,m,b' #let's test for a feature and add a slope 
    ch4h2_cld='ch4frac,cldp,offset'
    h2oh2_cld='h2ofrac,cldp,offset'
    co2h2_cld='co2frac,cldp,offset'
    mh_cld='logmh,cldp,offset'
    mh_cld_logf='logmh,cldp,offset,logf'
    ch4h2_cld_logf='ch4frac,cldp,offset,logf'
    h2oh2_cld_logf='h2ofrac,cldp,offset,logf'
    co2h2_cld_logf='co2frac,cldp,offset,logf'
    
class guesses_set: 
    """
    If you need to test your model it is helpful to have an initial guess of those parameters so that you can easily 
    test your model evaluation. If you are running emcee this could be used as an initial guess.
    """
    slope_0_line =[ 0.001293]
    line = [0,0.001293]
    step_line = [0.001293,0.001293]
    gauss_free = [3.4,  np.log10(0.05), -3, 0.001293]#'lam0,sig,Amp,cst'
    gauss_free_nrs1 = [3.4,  np.log10(0.05), -3, 0.001293,0.001293]
    gauss_free_nrs2 = [4.2,  np.log10(0.05), -3, 0.001293, 0.001293]
    gauss_ch4 = [3.13,  np.log10(0.05), -3, 0.001293,0.001293]
    gauss_line_free = [3.4, np.log10(0.05), -3, 0,0.001293] #'lam0,logsig,kogAmp,m,b'
    ch4h2_cld=[1,-1,0]
    h2oh2_cld=[1,-1,0]
    co2h2_cld=[1,-1,0]
    logmh_cld=[0,-1,0]
    
class model_set: 
    """
    Here are the models that should be associated with each of your parameters and guesses. 
    """
    def mh_cld(cube,wlgrid,plan=plan_og): 
        logMH=cube[0]
        log_cld_top = cube[1] #log units prior=[1.9 - -5]
        offset = cube[2]

        
        P = plan.inputs['atmosphere']['profile']['pressure'].values
        T = plan.inputs['atmosphere']['profile']['temperature'].values

        mc = MetallicityCalculator('zahnle_earth_ct.yaml')

        # Compute chemical equilibrium along profile
        CtoO = 1.0 # C/O ratio 1x solar (same as solar)
        #log10MH = 1.0 # metallicity = 10x solar
        df = mc.equilibrate(T, P, CtoO, logMH)

        df['temperature'] = T
        df['pressure'] = P
        plan.atmosphere(df=df)

        #add in the cloud
        cloud_bottom = 2 #100 bars
        log_dp = cloud_bottom-log_cld_top #2-1.9=0.1, 2+5 = 7 
        plan.clouds(g0=[0.9], w0=[0.9], opd=[10], p=[cloud_bottom], dp=[log_dp])

        #run picaso
        df_picaso = plan.spectrum(opas, calculation='transmission', full_output=True)
        x,y = df_picaso['wavenumber'], df_picaso['transit_depth']
        #flip and convert to um
        x = 1e4/x[::-1]
        y = y[::-1]
        biny = spectres.spectres(wlgrid, x ,y)

        biny = biny+offset
        
        return biny
    mh_cld_logf = mh_cld
    def ch4h2_cld(cube,wlgrid,plan=plan_og): 

        metal_to_fit = 'CH4'
        
        metalh2_fraction = 10**cube[0]
        log_cld_top = cube[1] #log units prior=[1.9 - -5]
        offset = cube[2]
        
        h2he = 1/(1+metalh2_fraction)
        metal = 1 - h2he

        h2he_frac = 0.854551/0.144022
        he = h2he/(1+h2he_frac)
        h2 = h2he-he
        
        plan.inputs['atmosphere']['profile'][metal_to_fit] = metal
        plan.inputs['atmosphere']['profile']['H2'] = h2
        plan.inputs['atmosphere']['profile']['He'] = he
        plan.atmosphere(df=plan.inputs['atmosphere']['profile'].loc[:,['pressure','temperature',metal_to_fit,'H2','He']])

        #add in the cloud
        cloud_bottom = 2 #100 bars
        log_dp = cloud_bottom-log_cld_top #2-1.9=0.1, 2+5 = 7 
        plan.clouds(g0=[0.9], w0=[0.9], opd=[10], p=[cloud_bottom], dp=[log_dp])

        #run picaso
        df_picaso = plan.spectrum(opas, calculation='transmission', full_output=True)
        x,y = df_picaso['wavenumber'], df_picaso['transit_depth']
        #flip and convert to um
        x = 1e4/x[::-1]
        y = y[::-1]
        biny = spectres.spectres(wlgrid, x ,y)

        biny = biny+offset
        
        return biny
    def h2oh2_cld(cube,wlgrid,plan=plan_og): 

        metal_to_fit = 'H2O'
        
        metalh2_fraction = 10**cube[0]
        log_cld_top = cube[1] #log units prior=[1.9 - -5]
        offset = cube[2]
        
        h2he = 1/(1+metalh2_fraction)
        metal = 1 - h2he

        h2he_frac = 0.854551/0.144022
        he = h2he/(1+h2he_frac)
        h2 = h2he-he

        plan.inputs['atmosphere']['profile'][metal_to_fit] = metal
        plan.inputs['atmosphere']['profile']['H2'] = h2
        plan.inputs['atmosphere']['profile']['He'] = he
        plan.atmosphere(df=plan.inputs['atmosphere']['profile'].loc[:,['pressure','temperature',metal_to_fit,'H2','He']])

        #add in the cloud
        cloud_bottom = 2 #100 bars
        log_dp = cloud_bottom-log_cld_top #2-1.9=0.1, 2+5 = 7 
        plan.clouds(g0=[0.9], w0=[0.9], opd=[10], p=[cloud_bottom], dp=[log_dp])

        #run picaso
        df_picaso = plan.spectrum(opas, calculation='transmission', full_output=True)
        x,y = df_picaso['wavenumber'], df_picaso['transit_depth']
        
        #flip and convert to um
        x = 1e4/x[::-1]
        y = y[::-1]
        biny = spectres.spectres(wlgrid, x ,y)
        biny = biny+offset
        
        return biny
    h2oh2_cld_logf=h2oh2_cld
    def co2h2_cld(cube,wlgrid,plan=plan_og): 

        metal_to_fit = 'CO2'
        
        metalh2_fraction = 10**cube[0]
        log_cld_top = cube[1] #log units prior=[1.9 - -5]
        offset = cube[2]
        
        h2he = 1/(1+metalh2_fraction)
        metal = 1 - h2he

        h2he_frac = 0.854551/0.144022
        he = h2he/(1+h2he_frac)
        h2 = h2he-he
        
        plan.inputs['atmosphere']['profile'][metal_to_fit] = metal
        plan.inputs['atmosphere']['profile']['H2'] = h2
        plan.inputs['atmosphere']['profile']['He'] = he
        plan.atmosphere(df=plan.inputs['atmosphere']['profile'].loc[:,['pressure','temperature',metal_to_fit,'H2','He']])

        #add in the cloud
        cloud_bottom = 2 #100 bars
        log_dp = cloud_bottom-log_cld_top #2-1.9=0.1, 2+5 = 7 
        plan.clouds(g0=[0.9], w0=[0.9], opd=[10], p=[cloud_bottom], dp=[log_dp])

        #run picaso
        df_picaso = plan.spectrum(opas, calculation='transmission', full_output=True)
        x,y = df_picaso['wavenumber'], df_picaso['transit_depth']
        #flip and convert to um
        x = 1e4/x[::-1]
        y = y[::-1]
        biny = spectres.spectres(wlgrid, x ,y)

        biny = biny+offset
        
        return biny
        
    def slope_0_line(cube,wlgrid):
        return 0*wlgrid + cube[0]
       
    def line(cube,wlgrid):
        return cube[0]*wlgrid + cube[1]
        
    def step_line(cube,wlgrid):
        spec = wlgrid*0
        spec[wlgrid<3.78] = spec[wlgrid<3.78] + cube[0]
        spec[wlgrid>=3.78] = spec[wlgrid>=3.78] + cube[1]
        return spec
        
    def gauss_free(cube,wlgrid):
        lam0, logsig, logAmp,cst = cube 
        sig = 10**logsig
        Amp = 10**logAmp
        val = (Amp*np.exp(-(wlgrid-lam0)**2/sig**2)+cst)
        return val
    
    def gauss_free_nrs1(cube,wlgrid):
        lam0, logsig, logAmp,cst_nr1,cst_nr2 = cube 
        sig = 10**logsig
        Amp = 10**logAmp
        val = (Amp*np.exp(-(wlgrid-lam0)**2/sig**2)+cst_nr1)
        val[wlgrid>=3.78] = cst_nr2
        return val

    def gauss_free_nrs2(cube,wlgrid):
        lam0, logsig, logAmp,cst_nr1,cst_nr2 = cube 
        sig = 10**logsig
        Amp = 10**logAmp
        val = (Amp*np.exp(-(wlgrid-lam0)**2/sig**2)+cst_nr2)
        val[wlgrid<3.78] = cst_nr1
        return val


    def gauss_ch4(cube,wlgrid):
        logsig, logAmp,cst1,cst2 = cube 
        lam0 = 3.314
        sig = 10**logsig
        Amp = 10**logAmp
        val = (Amp*np.exp(-(wlgrid-lam0)**2/sig**2)+cst1)
        val[wlgrid>=3.78] = cst2
        return val
    
    def gauss_line_free(cube,wlgrid):
        lam0,logsig, logAmp,m,b = cube 
        sig = 10**logsig
        Amp = 10**logAmp
        gauss = Amp*np.exp(-(wlgrid-lam0)**2/sig**2)
        line = wlgrid*m
        return gauss + line + b
    

class prior_set: 
    """
    And, for each model we need a prior bound for each of the gree parameters
    """
    def mh_cld(cube):
        params = cube.copy()
        min_mh = 0
        max_mh = 3
        params[0] = min_mh + (max_mh-min_mh)*params[0]

        min_cldtop = -5
        max_cldtop = 1.9
        params[1] = min_cldtop + (max_cldtop-min_cldtop)*params[1]

        min_offset = -100e-6
        max_offset = +100e-6
        params[2] = min_offset + (max_offset-min_offset)*params[2]
        return params
        
    def mh_cld_logf(cube):
        params = cube.copy()
        min_mh = 0
        max_mh = 3
        params[0] = min_mh + (max_mh-min_mh)*params[0]

        min_cldtop = -5
        max_cldtop = 1.9
        params[1] = min_cldtop + (max_cldtop-min_cldtop)*params[1]

        min_offset = -100e-6
        max_offset = +100e-6
        params[2] = min_offset + (max_offset-min_offset)*params[2]

        #logf
        minn = -2.5
        maxx = 2.5
        params[3] =  minn + (maxx-minn)*params[3]        
        return params
        
    def ch4h2_cld(cube):
        params = cube.copy()
        min_metal = -12
        max_metal = 2
        params[0] = min_metal + (max_metal-min_metal)*params[0]

        min_cldtop = -5
        max_cldtop = 1.9
        params[1] = min_cldtop + (max_cldtop-min_cldtop)*params[1]

        min_offset = -100e-6
        max_offset = +100e-6
        params[2] = min_offset + (max_offset-min_offset)*params[2]
        return params
        
    def ch4h2_cld_logf(cube):
        params = cube.copy()
        min_metal = -12
        max_metal = 2
        params[0] = min_metal + (max_metal-min_metal)*params[0]

        min_cldtop = -5
        max_cldtop = 1.9
        params[1] = min_cldtop + (max_cldtop-min_cldtop)*params[1]

        min_offset = -100e-6
        max_offset = +100e-6
        params[2] = min_offset + (max_offset-min_offset)*params[2]
        
        #logf
        minn = -7
        maxx = -3
        params[3] =  minn + (maxx-minn)*params[3]        
        return params
        
    #use same priors for everything
    h2oh2_cld = ch4h2_cld 
    co2h2_cld = ch4h2_cld
    h2oh2_cld_logf = ch4h2_cld_logf 
    co2h2_cld_logf = ch4h2_cld_logf    
    def slope_0_line(cube):
        params = cube.copy()
        minv = -5#.0001
        maxv = 5#.005
        params[0] = minv + (maxv-minv)*params[0]
        return params

    def step_line(cube):
        params = cube.copy()
        minv = -5#.0001
        maxv = 5#.005
        params[0] = minv + (maxv-minv)*params[0]
        minv = -5#.0001
        maxv = 5#.005
        params[1] = minv + (maxv-minv)*params[1]
        return params    
    
    def line(cube):
        params = cube.copy()
        minv = -5#m
        maxv = 5
        params[0] = minv + (maxv-minv)*params[0]
        minv = -5#b
        maxv = 5
        params[1] = minv + (maxv-minv)*params[1]
        return params

    def gauss_free(cube):  
        params = cube.copy()
        lam0, logsig, logAmp,cst=params
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
        cst = minv + (maxv-minv)*cst
        params =[lam0, logsig, logAmp,cst]
        return params 

    def gauss_free_nrs1(cube):  
        params = cube.copy()
        lam0, logsig, logAmp,cst_nrs1,cst_nrs2 =params
        mina =-2
        maxa =1.5 
        logAmp=mina+(maxa-mina)*logAmp
        min_wavelength=3
        max_wavelength=3.78
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
        
    def gauss_ch4(cube):  
        params = cube.copy()
        logsig, logAmp,cst_nrs1,cst_nrs2 =params
        mina =-2
        maxa =1.5 
        logAmp=mina+(maxa-mina)*logAmp
        
        min_width = np.log10(0.01)
        max_width = np.log10(2)
        logsig=min_width+(max_width-min_width)*logsig
        
        minv = -5
        maxv = 5
        cst_nrs1 = minv + (maxv-minv)*cst_nrs1
        
        minv = -5
        maxv = 5
        cst_nrs2 = minv + (maxv-minv)*cst_nrs2
        params =[logsig, logAmp,cst_nrs1, cst_nrs2]
        return params
        
    def gauss_free_nrs2(cube):  
        params = cube.copy()
        lam0, logsig, logAmp, cst_nrs1, cst_nrs2=params
        mina =-2
        maxa =1.5 
        logAmp=mina+(maxa-mina)*logAmp
        min_wavelength=3.78
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
        
    def gauss_line_free(cube):
        params = cube.copy()
        lam0, logsig, logAmp,m,b=params
        mina =-2
        maxa =1.5 
        logAmp=mina+(maxa-mina)*logAmp
        min_wavelength=3
        max_wavelength=5.2
        lam0=min_wavelength+(max_wavelength-min_wavelength)*lam0 
        min_width = np.log10(0.01)
        max_width = np.log10(2)
        logsig=min_width+(max_width-min_width)*logsig
        #line
        minv = -5#m
        maxv =5
        m = minv + (maxv-minv)*m
        minv = -5#b
        maxv = 5
        b = minv + (maxv-minv)*b
        params =[lam0, logsig, logAmp,m,b]
        return params 
    

def model_double_gauss(wlgrid, lam01, sig1, Amp1,lam02, sig2, Amp2,cst2):
        return ((Amp1*np.exp(-(wlgrid-lam01)**2/sig1**2)) + 
                (Amp2*np.exp(-(wlgrid-lam02)**2/sig2**2))  + cst2  )    