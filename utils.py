import numpy as np
from scipy import constants as const
import pandas as pd
import yaml
from tempfile import NamedTemporaryFile
from spectres.spectral_resampling import make_bins

from photochem.clima import rebin
from photochem import equilibrate, zahnle_earth

###
### Misc utils
###

def equilibrium_temperature(stellar_radiation, bond_albedo):
    T_eq = ((stellar_radiation*(1.0 - bond_albedo))/(4.0*const.sigma))**(0.25)
    return T_eq 

def residuals(data_y, err, expected_y):
    return (data_y - expected_y)/err

def chi_squared(data_y, err, expected_y):
    R = residuals(data_y, err, expected_y)
    return np.sum(R**2)

def get_data(fname):
    if 'JA' in fname:
        df = pd.read_csv(fname, skiprows=1, header=None, names=['num','um','bin','tdepth', 'edepth'])
        x,xbhw,y,e = (
            df['um'].values, 
            df['bin'].values, 
            df['tdepth'].values,
            df['edepth'].values,
        )
    elif "NW" in fname:
        df = pd.read_csv(fname, skiprows=1, header=None, names=['um', 'tdepth', 'edepth'])
        x,y,e = (df['um'].values, 
                df['tdepth'].values,
                df['edepth'].values,
                )
        # Make separate bins for NRS1 and NRS2
        _, xbhw1 = make_bins(x[x<3.78])
        _, xbhw2 = make_bins(x[x>=3.78])
        xbhw = np.append(xbhw1,xbhw2)
        xbhw = xbhw/2.0
        # xbhw = np.ones(x.shape[0])*(x[1] - x[0])/2
    else:
        raise Exception()
    return x,xbhw,y,e

def get_data_dict(fname):
    wv, wv_bin_hw, rprs2, rprs2_err = get_data(fname)
    tmp = {}
    tmp['wv'] = wv
    tmp['wv_bin_hw'] = wv_bin_hw
    tmp['rprs2'] = rprs2
    tmp['rprs2_err'] = rprs2_err
    return tmp

def rebin_picaso_to_data(wv_model, flux_model, wv_data, wv_data_bin_halfwidth):
    "Rebins picaso model output to data."
    wv = wv_model.copy()
    flux_vals = flux_model.copy()
    wv_data = wv_data.copy()
    wv_data_bin_halfwidth = wv_data_bin_halfwidth.copy()

    # Get bins of data
    wv_bins_data = np.empty((wv_data.shape[0],2))
    wv_bins_data[:,0] = wv_data - wv_data_bin_halfwidth
    wv_bins_data[:,1] = wv_data + wv_data_bin_halfwidth

    # Get bins of Picaso
    d = np.diff(wv)
    wv_bins = np.array([wv[0]-d[0]/2] + list(wv[0:-1]+d/2.0) + [wv[-1]+d[-1]/2]).copy()

    flux_vals_new = np.empty(wv_bins_data.shape[0])
    for i in range(wv_bins_data.shape[0]):
        flux_vals_new[i] = rebin(wv_bins, flux_vals, wv_bins_data[i,:].copy())[0]

    return wv_bins_data, flux_vals_new

###
### Chemical equilibrium solver
###

def generate_zahnle_earth_thermo():
    """Generates zahnle thermodynamic dictionary for chemical equilibrium solver

    Returns
    -------
    dict
        Dictionary with information needed for chemical equilibrium solver
    """     

    with open(zahnle_earth,'r') as f:
        dat = yaml.load(f, Loader=yaml.Loader)

    with open(zahnle_earth.replace('zahnle_earth.yaml','condensate_thermo.yaml'),'r') as f:
        dat1 = yaml.load(f, Loader=yaml.Loader)

    # Delete information that is not needed
    for i,atom in enumerate(dat['atoms']):
        del dat['atoms'][i]['redox'] 
    del dat['particles']
    del dat['reactions']

    for i,sp in enumerate(dat1['species']):
        dat['species'].append(sp)

    return dat

class MetalicityCalculator():
    "A simple Metallicity calculator." 

    def __init__(self, thermofile=None):
        """Initializes the model.

        Parameters
        ----------
        thermofile : str, optional
            Input thermodynamic file, by default None.
        """        

        if thermofile is None:
            # If no thermo file, then use Zahnle Earth.
            with NamedTemporaryFile('w',suffix='.yaml') as f:
                yaml.dump(generate_zahnle_earth_thermo(), f, Dumper=yaml.Dumper)
                self.gas = equilibrate.ChemEquiAnalysis(f.name)
        else:
            self.gas = equilibrate.ChemEquiAnalysis(thermofile)

    def solve(self, T, P, CtoO, metal, rainout_condensed_atoms=True):
        """Given a T-P profile, C/O ratio and metallicity, the code
        computes chemical equilibrium composition.

        Parameters
        ----------
        T : ndarray[dim=1,float64]
            Temperature in K
        P : ndarray[dim=1,float64]
            Pressure in dynes/cm^2
        CtoO : float
            The C / O ratio relative to solar. CtoO = 1 would be the same
            composition as solar.
        metal : float
            Metallicity relative to solar.
        rainout_condensed_atoms : bool, optional
            If True, then the code will rainout atoms that condense.

        Returns
        -------
        dict
            Composition at chemical equilibrium.
        """

        # Check T and P
        if isinstance(T, float) or isinstance(T, int):
            T = np.array([T],np.float64)
        if isinstance(P, float) or isinstance(P, int):
            P = np.array([P],np.float64)
        if not isinstance(P, np.ndarray):
            raise ValueError('"P" must by an np.ndarray')
        if not isinstance(T, np.ndarray):
            raise ValueError('"P" must by an np.ndarray')
        if T.ndim != 1:
            raise ValueError('"T" must have one dimension')
        if P.ndim != 1:
            raise ValueError('"P" must have one dimension')
        if T.shape[0] != P.shape[0]:
            raise ValueError('"P" and "T" must be the same length')
        if not np.all(np.diff(P) < 0):
            raise ValueError('"P" must be decreasing with height')
        # Check CtoO and metal
        if CtoO <= 0:
            raise ValueError('"CtoO" must be greater than 0')
        if metal <= 0:
            raise ValueError('"metal" must be greater than 0')
        
        molfracs_atoms = self.gas.molfracs_atoms_sun
        for i,sp in enumerate(self.gas.atoms_names):
            if sp != 'H' and sp != 'He':
                molfracs_atoms[i] = self.gas.molfracs_atoms_sun[i]*metal
        molfracs_atoms = molfracs_atoms/np.sum(molfracs_atoms)

        # Adjust C and O to get desired C/O ratio. CtoO is relative to solar
        indC = self.gas.atoms_names.index('C')
        indO = self.gas.atoms_names.index('O')
        x = CtoO*(molfracs_atoms[indC]/molfracs_atoms[indO])
        a = (x*molfracs_atoms[indO] - molfracs_atoms[indC])/(1+x)
        molfracs_atoms[indC] = molfracs_atoms[indC] + a
        molfracs_atoms[indO] = molfracs_atoms[indO] - a

        # For output
        out = {}
        for sp in self.gas.gas_names:
            out[sp] = np.empty(P.shape[0])

        # Compute chemical equilibrium at all alititudes
        for i in range(P.shape[0]):
            self.gas.solve(P[i], T[i], molfracs_atoms=molfracs_atoms)
            for j,sp in enumerate(self.gas.gas_names):
                out[sp][i] = self.gas.molfracs_species_gas[j]
            if rainout_condensed_atoms:
                molfracs_atoms = self.gas.molfracs_atoms_gas

        return out
    
    def solve_picaso(self, T, P, CtoO, log10metal, rainout_condensed_atoms=True):
        """Uses PICASO conventions for inputs/outputs, such that P vs. T is flipped.

        Parameters
        ----------
        T : ndarray[dim=1,float64]
            Temperature in K
        P : ndarray[dim=1,float64]
            Pressure in bars. First element is the top of the atmosphere.
        CtoO : float
            The C / O ratio relative to solar. CtoO = 1 would be the same
            composition as solar.
        log10metal : float
            log10 metallicity relative to solar.
        rainout_condensed_atoms : bool, optional
            If True, then the code will rainout atoms that condense.

        Returns
        -------
        DataFrame
            Composition at chemical equilibrium.
        """        

        out = self.solve(np.flip(T).copy(), np.flip(P).copy()*1.0e6, CtoO, 10.0**log10metal, rainout_condensed_atoms)
        for key in out:
            out[key] = np.flip(out[key]).copy()

        return pd.DataFrame(out)