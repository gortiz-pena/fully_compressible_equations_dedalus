import logging
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

class Polytrope():
    
    def __init__(self, n_rho, epsilon, gamma=5./3):
        self.m_ad    = 1/(gamma-1)
        self.m       = self.m_ad - epsilon
        self.epsilon = epsilon
        self.n_rho   = n_rho
        self.Lz      = np.exp(self.n_rho/self.m) - 1
        self.gamma   = gamma
    
    def build_atmosphere(self, domain, problem):
        self.domain = domain
        self.problem = problem

        # Construct atmosphere
        self.T0        = domain.new_field()
        self.T0_z      = domain.new_field()
        self.rho0      = domain.new_field()
        self.ln_rho0_z = domain.new_field()
        for f in [self.T0, self.T0_z, self.rho0, self.ln_rho0_z]:
            if domain.dim > 1:
                f.meta['x']['constant'] = True
            if domain.dim == 3:
                f.meta['y']['constant'] = True
            f.set_scales(self.domain.dealias)

        z_de = domain.grid(-1, scales=self.domain.dealias)

        self.T0['g']   = 1 + self.Lz - z_de
        self.rho0['g'] = self.T0['g']**self.m
        self.T0.differentiate('z', out=self.T0_z)
        self.rho0.differentiate('z', out=self.ln_rho0_z)
        self.ln_rho0_z['g'] /= self.rho0['g']


        # Atmosphere
        self.problem.parameters['T0']                  = self.T0
        self.problem.parameters['T0_z']                = self.T0_z
        self.problem.parameters['rho0']                = self.rho0
        self.problem.parameters['ln_rho0_z']           = self.ln_rho0_z

        # Nondimensionalization
        self.problem.parameters['g']  = self.g  = 1 + self.m
        self.problem.parameters['R']  = self.R  = 1
        self.problem.parameters['ɣ']  = self.ɣ  = self.gamma
        self.problem.parameters['Cp'] = self.Cp = self.R * self.gamma / (self.gamma-1)
        self.problem.parameters['Cv'] = self.Cv = self.Cp - self.R

        self.ds_dz_over_cp = domain.new_field()
        self.ds_dz_over_cp.set_scales(domain.dealias)
        self.ds_dz_over_cp['g'] =  (1/self.ɣ)*self.T0_z['g']/self.T0['g'] - (self.ɣ-1)/self.ɣ * self.ln_rho0_z['g']

        self.problem.substitutions['m_scale'] = self.problem.substitutions['c_scale'] = 'T0'
        self.problem.substitutions['e_scale'] = '1'
        self.problem.parameters['rho0_min']   = np.mean(self.rho0.interpolate(z=self.Lz)['g'])
        self.problem.substitutions['grav_phi'] = '(-g*(1 + Lz - z))'

        self.problem.parameters['Lz'] = self.Lz

    def set_parameters(self, Ra=1, Pr=1, aspect=2, Pm=None, Ta=None):
        """ Given the Rayleigh and Taylor numbers are defined at the top of the domain """
        self.problem.parameters['Lx'] = self.problem.parameters['Ly'] = aspect*self.Lz

       
        delta_s_over_cp = np.mean(self.ds_dz_over_cp.integrate('z')['g'])
        K  = self.Cp*np.sqrt(self.g*self.Lz**3*np.abs(delta_s_over_cp) / Ra / Pr)
        μ  = K * Pr / self.Cp

        self.problem.parameters['K']     = K 
        self.problem.parameters['μ']     = μ

        logger.info("Ra = {:.3e}, Pr = {:2g}".format(Ra, Pr))
        logger.info("K = {:.3e}, η = {:2e}".format(K, μ))

        if Pm is not None:
            η  = μ * Pm 
            self.problem.parameters['η']     = η 
            self.problem.parameters['μ0']    = self.μ0 = 1
            self.problem.substitutions['ohm_scale']         = 'μ0*η'
            logger.info("Pm = {:.3e}, μ = {:2e}".format(Pm, μ))
        if Ta is not None:
            Ω0 = (μ / 2 / self.Lz**2) * np.sqrt(Ta)
            self.problem.parameters['Ω0']    = Ω0 
            self.problem.substitutions['φ']  = '0'
            logger.info("Ta = {:.3e}, Ω0 = {:.3e}".format(Ta, Ω0))


        self.t_buoy  = np.sqrt(self.g*self.Lz/np.abs(delta_s_over_cp) )
        self.t_diff  = np.sqrt(self.Lz/μ)
        return self.t_buoy, self.t_diff


