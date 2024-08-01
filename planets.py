
class Star:
    radius : float # relative to the sun
    Teff : float # K
    metal : float # log10(M/H)
    kmag : float
    logg : float
    planets : dict # dictionary of planet objects

    def __init__(self, radius, Teff, metal, kmag, logg, planets):
        self.radius = radius
        self.Teff = Teff
        self.metal = metal
        self.kmag = kmag
        self.logg = logg
        self.planets = planets
        
class Planet:
    radius : float # in Earth radii
    mass : float # in Earth masses
    Teq : float # Equilibrium T in K
    transit_duration : float # in seconds
    a: float # semi-major axis in AU
    a: float # semi-major axis in AU
    stellar_flux: float # W/m^2
    
    def __init__(self, radius, mass, Teq, transit_duration, a, stellar_flux):
        self.radius = radius
        self.mass = mass
        self.Teq = Teq
        self.transit_duration = transit_duration
        self.a = a
        self.stellar_flux = stellar_flux

# GJ 357 b
TOI562_01 = Planet(
    radius=1.217, # Luque et al. (2019), A&A
    mass=1.84, # Luque et al. (2019), A&A
    Teq=525, # Luque et al. (2019), A&A
    transit_duration=1.53*60*60, # Exo.MAST
    a=0.035, # Luque et al. (2019), A&A
    stellar_flux=12.6*1368 # Luque et al. (2019), A&A
)

TOI562 = Star(
    radius=0.337, # Luque et al. (2019), A&A
    Teff=3505, # Luque et al. (2019), A&A
    metal=-0.12, # Luque et al. (2019), A&A
    kmag=6.475, # Luque et al. (2019), A&A
    logg=4.94, # Luque et al. (2019), A&A
    planets={'01':TOI562_01}
)








