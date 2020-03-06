import logging
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

class LinearAtmosphere():
    
    def __init__(self, domain, problem):
        self.domain = domain
        self.problem = problem

        # Construct atmosphere
        self.T0        = domain.new_field()
        self.T0_z      = domain.new_field()
        self.rho0      = domain.new_field()
        self.ln_rho0_z = domain.new_field()
        for f in [self.T0, self.T0_z, self.rho0, self.ln_rho0_z]:
            f.meta['x']['constant'] = True
            if domain.dim == 3:
                f.meta['y']['constant'] = True

        z = domain.grid(-1)

        self.T0['g']   = (5/12)*(4 - 3*z)
        self.rho0['g'] = (2/ 5)*(4 - 3*z)
        self.T0.differentiate('z', out=self.T0_z)
        self.rho0.differentiate('z', out=self.ln_rho0_z)
        self.ln_rho0_z['g'] /= self.rho0['g']

        # Atmosphere
        problem.parameters['T0']                  = self.T0
        problem.parameters['T0_z']                = self.T0_z
        problem.parameters['rho0']                = self.rho0
        problem.parameters['ln_rho0_z']           = self.ln_rho0_z

        # Nondimensionalization
        problem.parameters['g']  = self.g  = 1
        problem.parameters['Cp'] = self.Cp = 1
        problem.parameters['Cv'] = self.Cv = 3/5
        problem.parameters['ɣ']  = self.ɣ  = 5/3

        self.ds_dz_over_cp = domain.new_field()
        self.ds_dz_over_cp.set_scales(domain.dealias)
        self.ds_dz_over_cp['g'] =  (1/self.ɣ)*self.T0_z['g']/self.T0['g'] - (1-self.ɣ)/self.ɣ * self.ln_rho0_z['g']

    def set_parameters(self, Ra=1, Pr=1, aspect=2, Pm=None, Ta=None):

        
        self.problem.parameters['Lx'] = self.problem.parameters['Ly'] = aspect
        self.problem.parameters['Lz'] = self.d = 1

        rho_m   = np.mean(self.rho0.interpolate(z=0.5)['g'])
        ds_dz_m = np.mean(self.ds_dz_over_cp.interpolate(z=0.5)['g'])
        K  = rho_m*self.Cp*np.sqrt(self.g*self.d**4*rho_m**2 * self.Cp * -1 * ds_dz_m / Ra / Pr)
        μ  = K * Pr / self.Cp

        self.problem.parameters['K']     = K 
        self.problem.parameters['μ']     = μ
        self.problem.substitutions['R']                 = '(ɣ-1)*Cv'
        self.problem.substitutions['visc_scale']        = 'μ'
        self.problem.substitutions['cond_scale']        = '(K)'

        logger.info("Ra = {:.3e}, Pr = {:2g}".format(Ra, Pr))
        logger.info("K = {:.3e}, η = {:2e}".format(K, μ))

        if Pm is not None:
            η  = μ * Pm / rho_m
            self.problem.parameters['η']     = η 
            self.problem.parameters['μ0'] = self.μ0 = 1
            self.problem.substitutions['ohm_scale']         = 'μ0*η'
            logger.info("Pm = {:.3e}, μ = {:2e}".format(Pm, μ))
        if Ta is not None:
            Ω0 = (μ / rho_m / 2 / d**2) * np.sqrt(Ta)
            self.problem.parameters['Ω0']    = Ω0 
            self.problem.substitutions['coriolis_scale']    = '2*Ω0'
            logger.info("Ta = {:.3e}, Ω0 = {:.3e}".format(Ta, Ω0))


        delta_S = np.abs(np.mean(self.ds_dz_over_cp.integrate()['g'])*self.Cp)
        self.t_buoy  = np.sqrt(self.g*self.Cp*self.d/delta_S) 
        self.t_diff  = np.sqrt(self.d/μ)
        return self.t_buoy, self.t_diff


