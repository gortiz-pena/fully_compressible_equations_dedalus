import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

class FCMHDEquations():
    
    def __init__(self):
        self.variables       = ['T1', 'T1_z', 'ln_rho1', 'u', 'v', 'w', 'u_z', 'v_z', 'w_z', 'Bx', 'By', 'Bz', 'Ax', 'Ay', 'Az', 'phi']
        self.necessary_terms = ['R', 'visc_scale', 'cond_scale', 'ohm_scale', 'coriolis_scale', 'ɣ', 'g', 'Cv', 'Cp', 'T0', 'T0_z', 'rho0', 'ln_rho0_z']
        self._define_equations()
        self._define_BCs()

        self.subs = OrderedDict()
        self._define_operators()
        self._define_physical_subs()
        self._define_output_subs()


    def _define_equations(self):
        self.equations  = OrderedDict()
        self.equations['Bx_defn']       = "Bx + dz(Ay) - dy(Az) = 0"
        self.equations['By_defn']       = "By + dx(Az) - dz(Ax) = 0"
        self.equations['Bz_defn']       = "Bz + dy(Ax) - dx(Ay) = 0"
        self.equations['T1_z_fof']      = "dz(T1) - T1_z = 0"
        self.equations['u_z_fof']       = "dz(u)  - u_z  = 0"
        self.equations['v_z_fof']       = "dz(v)  - v_z  = 0"
        self.equations['w_z_fof']       = "dz(w)  - w_z  = 0"
        self.equations['Continuity']    = "T0*(dt(ln_rho1) + w*ln_rho0_z + DivU) = -T0*UdotGrad(ln_rho1, dz(ln_rho1))"
        self.equations['Induction_x']   = "dt(Ax) + μ0*η*Jx + dx(phi) = v*Bz - w*By"
        self.equations['Induction_y']   = "dt(Ay) + μ0*η*Jy + dy(phi) = w*Bx - u*Bz"
        self.equations['Induction_z']   = "dt(Az) + μ0*η*Jz + dz(phi) = u*By - v*Bx"
        self.equations['Momentum_x']    = "T0*(dt(u)  + R*(dx(T1) + T0*dx(ln_rho1))                - coriolis_scale*v - visc_u_L) = T0*(-UdotGrad(u, u_z) - R*T1*dx(ln_rho1) + (1/rho_full)*(Jy*Bz - Jz*By) + visc_u_R)"
        self.equations['Momentum_y']    = "T0*(dt(v)  + R*(dy(T1) + T0*dy(ln_rho1))                + coriolis_scale*u - visc_v_L) = T0*(-UdotGrad(v, v_z) - R*T1*dy(ln_rho1) + (1/rho_full)*(Jz*Bx - Jx*Bz) + visc_v_R)"
        self.equations['Momentum_z']    = "T0*(dt(w)  + R*(T1_z   + T0*dz(ln_rho1) + T1*ln_rho0_z)                    - visc_w_L) = T0*(-UdotGrad(w, w_z) - R*T1*dz(ln_rho1) + (1/rho_full)*(Jx*By - Jy*Bx) + visc_w_R)"
        self.equations['Energy']        = "T0*(dt(T1) + w*T0_z + (ɣ-1)*T0*DivU - 2*cond_scale*(1/ɣ)*Lap(T1, T1_z)/rho0                ) = T0*(-UdotGrad(T1, T1_z) - (ɣ-1)*T1*DivU + cond_scale*(1/ɣ)*Lap(T1, T1_z)*(1/rho_full - 2/rho0) + visc_heat + ohm_heat)"
        self.equations['Coulomb_gauge'] = "Div(Ax, Ay, dz(Az)) = 0"

    def _define_BCs(self):
        self.BCs = OrderedDict()
        # velocity
        self.BCs['stressfree_u_L'] = (" left(u_z) = 0", "True")
        self.BCs['stressfree_u_R'] = ("right(u_z) = 0", "True")
        self.BCs['stressfree_v_L'] = (" left(v_z) = 0", "True")
        self.BCs['stressfree_v_R'] = ("right(v_z) = 0", "True")
        self.BCs['noslip_u_L']     = (" left(u) = 0"  , "True")
        self.BCs['noslip_u_R']     = ("right(u) = 0"  , "True")
        self.BCs['noslip_v_L']     = (" left(v) = 0"  , "True")
        self.BCs['noslip_v_R']     = ("right(v) = 0"  , "True")
        self.BCs['impenetrable_L'] = (" left(w) = 0"  , "True")
        self.BCs['impenetrable_R'] = ("right(w) = 0"  , "True")
        
        #B field
        self.BCs['noHorizB_Ax_L']  = (" left(dz(Ax)) = 0", "True")
        self.BCs['noHorizB_Ax_R']  = ("right(dz(Ax)) = 0", "True")
        self.BCs['noHorizB_Ay_L']  = (" left(dz(Ay)) = 0", "True")
        self.BCs['noHorizB_Ay_R']  = ("right(dz(Ay)) = 0", "True")
        self.BCs['noHorizB_Az_L']  = (" left(Az ) = 0",    "True")
        self.BCs['noHorizB_Az_R1'] = ("right(Az ) = 0",    "(nx != 0) or (ny != 0)")
        self.BCs['noHorizB_Az_R2'] = ("right(phi) = 0",    "(nx == 0) and (ny == 0)")

        #Temperature
        self.BCs['temp_L']         = (" left(T1) = 0",   "True")
        self.BCs['temp_R']         = ("right(T1) = 0",   "True")
        self.BCs['flux_L']         = (" left(T1_z) = 0", "True")
        self.BCs['flux_R']         = ("right(T1_z) = 0", "True")

    def define_subs(self, problem):
        for k in self.necessary_terms:
            if k not in problem.substitutions.keys() and k not in problem.parameters.keys():
                logger.error("{} not specified! Need to specify before equations can be constructed.")
        for key, sub in self.subs.items():
            problem.substitutions[key] = sub
        return problem

    def _define_operators(self):
        self.subs['Lap(A, A_z)']       = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
        self.subs['UdotGrad(A, A_z)']  = '(u*dx(A) + v*dy(A) + w*A_z)'
        self.subs['Div(Ax, Ay, Az_z)'] = '(dx(Ax) + dy(Ay) + Az_z)'
        self.subs['DivU']              = 'Div(u, v, w_z)'
        self.subs['plane_avg(A)']      = 'integ(A, "x", "y")/Lx/Ly'
        self.subs['vol_avg(A)']        = 'integ(A)/Lx/Ly/Lz'
        self.subs['plane_std(A)']      = 'sqrt(plane_avg((A - plane_avg(A))**2))'

    def _define_physical_subs(self):
        self.subs['ln_rho0']          = '(log(rho0))'
        self.subs['rho_full']         = '(rho0*exp(ln_rho1))'
        self.subs['T_full']           = '(T0 + T1)'
            
        self.subs['Jx'] = '(dy(Bz) - dz(By))'
        self.subs['Jy'] = '(dz(Bx) - dx(Bz))'
        self.subs['Jz'] = '(dx(By) - dy(Bx))'

        # Stress Tensor and other hard terms
        self.subs["Sig_xx"] = "(2*dx(u) - 2/3*DivU)"
        self.subs["Sig_yy"] = "(2*dy(v) - 2/3*DivU)"
        self.subs["Sig_zz"] = "(2*w_z   - 2/3*DivU)"
        self.subs["Sig_xy"] = "(dx(v) + dy(u))"
        self.subs["Sig_xz"] = "(dx(w) +  u_z )"
        self.subs["Sig_yz"] = "(dy(w) +  v_z )"

        self.subs['visc_u']   = "( Lap(u, u_z) + 1/3*dx(DivU) )"
        self.subs['visc_v']   = "( Lap(v, v_z) + 1/3*dy(DivU) )"
        self.subs['visc_w']   = "( Lap(w, w_z) + 1/3*Div(u_z, v_z, dz(w_z)) )"

        self.subs['visc_u_L'] = 'visc_scale*visc_u*(1/rho0)'
        self.subs['visc_v_L'] = 'visc_scale*visc_v*(1/rho0)'
        self.subs['visc_w_L'] = 'visc_scale*visc_w*(1/rho0)'
        self.subs['visc_u_R'] = 'visc_scale*visc_u*(1/rho_full - 1/rho0)'
        self.subs['visc_v_R'] = 'visc_scale*visc_v*(1/rho_full - 1/rho0)'
        self.subs['visc_w_R'] = 'visc_scale*visc_w*(1/rho_full - 1/rho0)'

        self.subs['visc_heat']  = "visc_scale*(dx(u)*Sig_xx + dy(v)*Sig_yy + w_z*Sig_zz + Sig_xy**2 + Sig_xz**2 + Sig_yz**2)"
        self.subs['ohm_heat']   = 'ohm_scale*(Jx**2 + Jy**2 + Jz**2)'


    def _define_output_subs(self):
        # Velocity defns
        self.subs['u_rms'] = 'sqrt(u**2)'
        self.subs['v_rms'] = 'sqrt(v**2)'
        self.subs['w_rms'] = 'sqrt(w**2)'
        self.subs['vel_rms'] = 'sqrt(u**2 + v**2 + w**2)'
        self.subs['u_perp']  = 'sqrt(u**2 + v**2)'
        self.subs['Vort_x'] = '(dy(w) - v_z)'
        self.subs['Vort_y'] = '( u_z  - dx(w))'
        self.subs['Vort_z'] = '(dx(v) - dy(u))'
        self.subs['enstrophy']   = '(Vort_x**2 + Vort_y**2 + Vort_z**2)'

        # Mag defns
        self.subs['Bx_rms'] = 'sqrt(Bx**2)'
        self.subs['By_rms'] = 'sqrt(By**2)'
        self.subs['Bz_rms'] = 'sqrt(Bz**2)'
        self.subs['B_rms']  = 'sqrt(Bx**2 + By**2 + Bz**2)'
        self.subs['B_perp'] = 'sqrt(Bx**2 + By**2)'

        #thermo
        self.subs['s_over_cp'] = '((1/ɣ)*log(T_full) - ((ɣ-1)/ɣ)*log(rho_full))'
        self.subs['P_full']    = 'R*rho_full*T_full'

        #Diffusivities
        self.subs['nu']        = 'visc_scale/rho_full'
        self.subs['chi']       = 'visc_scale/(rho_full*Cp)'
        self.subs['Re_rms']    = 'vel_rms*Lz/nu'
        self.subs['Pe_rms']    = 'vel_rms*Lz/chi'
        self.subs['Ma_rms']    = 'vel_rms/sqrt(R*T_full)'

        #Energies
        self.subs['KE']        = 'rho_full*vel_rms**2/2'
        self.subs['IE']        = 'rho_full*Cv*T_full'
        self.subs['PE']        = 'rho_full*g*(z-1)'
        self.subs['BE']        = 'B_rms/2'
        self.subs['PE_fluc']   = '(PE - rho0*g*(z-1))'
        self.subs['IE_fluc']   = '(IE - rho0*Cv*T0)'
        self.subs['TE']        = '(KE + IE + PE + BE)'
        self.subs['TE_fluc']   = '(KE + BE + IE_fluc + PE_fluc)'

        #Fluxes
        self.subs['ohm_flux_z']   = 'ohm_scale*(Jx*By - Jy*Bx)'
        self.subs['poynt_flux_z'] = '(-(u*Bx + v*By)*Bz + (Bx**2 + By**2)*w)'
        self.subs['enth_flux_z']  = '(w*(IE+P_full))'
        self.subs['KE_flux_z']    = '(w*KE)'
        self.subs['PE_flux_z']    = '(w*PE)'
        self.subs['visc_flux_z']  = '(-visc_scale * (u*Sig_xz + v*Sig_yz + w*Sig_zz))'
        self.subs['F_cond_z']     = '(-cond_scale * dz(T_full))'
        self.subs['F_cond0_z']    = '(-cond_scale * dz(T0))'
        self.subs['F_cond1_z']    = '(-cond_scale * dz(T1))'



class FCEquations3D(FCMHDEquations):

    def __init__(self):
        super(FCEquations3D, self).__init__()
        self.unneeded_variables  = ['Bx', 'By', 'Bz', 'Ax', 'Ay', 'Az', 'phi']
        self.unnecessary_terms   = ['ohm_scale',] 
        self.unnecessary_subs    = ['Jx', 'Jy', 'Jz', 'ohm_heat', 'Bx_rms', 'By_rms', 'Bz_rms', 'B_rms', 'B_perp', 'BE', 'ohm_flux_z', 'poynt_flux_z']
        for k in self.unneeded_variables:
            if k in self.variables: self.variables.remove(k)

    def _define_equations(self):
        super(FCEquations3D, self)._define_equations()
        remove_eqns = ['Bx_defn', 'By_defn', 'Bz_defn', 'Induction_x', 'Induction_y', 'Induction_z', 'Coulomb_gauge']
        for k in remove_eqns:
            self.equations.pop(k)

    def _define_BCs(self):
        super(FCEquations3D, self)._define_BCs()
        remove_keys = ['noHorizB',]
        true_remove_keys = []
        for k in self.BCs.keys():
            for rk in remove_keys:
                if rk in k:
                    true_remove_keys.append(k)
        for k in true_remove_keys:
            self.BCs.pop(k)

    def define_subs(self, problem):
        for k in self.unneeded_variables:
            problem.parameters[k] = 0
        for k in self.unnecessary_terms:
            self.necessary_terms.remove(k)
            problem.substitutions[k] = '0'
        for k in self.unnecessary_subs:
            self.subs.pop(k)
            problem.substitutions[k] = '0'
        problem = super(FCEquations3D, self).define_subs(problem)
        return problem

class FCEquations2D(FCEquations3D):

    def __init__(self):
        super(FCEquations2D, self).__init__()
        self.unneeded_variables  += ['v', 'v_z']
        self.unnecessary_terms += ['coriolis_scale']
        for k in self.unneeded_variables:
            if k in self.variables: self.variables.remove(k)

    def _define_operators(self):
        super(FCEquations2D, self)._define_operators()
        self.subs.pop('plane_avg(A)')
        self.subs.pop('vol_avg(A)')
        self.subs['plane_avg(A)']      = 'integ(A, "x")/Lx'
        self.subs['vol_avg(A)']        = 'integ(A)/Lx/Lz'

    def _define_equations(self):
        super(FCEquations2D, self)._define_equations()
        remove_eqns = ['v_z_fof', 'Momentum_y']
        for k in remove_eqns:
            self.equations.pop(k)

    def _define_BCs(self):
        super(FCEquations2D, self)._define_BCs()
        remove_keys = ['stressfree_v', 'noslip_v']
        true_remove_keys = []
        for k in self.BCs.keys():
            for rk in remove_keys:
                if rk in k:
                    true_remove_keys.append(k)
        for k in true_remove_keys:
            self.BCs.pop(k)

    def define_subs(self, problem):
        problem.substitutions['dy(A)'] = 0
        problem = super(FCEquations2D, self).define_subs(problem)
        return problem
