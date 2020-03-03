import numpy as np
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict

def define_output_subs(problem):
    # Velocity defns
    problem.substitutions['u_rms'] = 'sqrt(u**2)'
    problem.substitutions['v_rms'] = 'sqrt(v**2)'
    problem.substitutions['w_rms'] = 'sqrt(w**2)'
    problem.substitutions['vel_rms'] = 'sqrt(u**2 + v**2 + w**2)'
    problem.substitutions['u_perp']  = 'sqrt(u**2 + v**2)'
    problem.substitutions['Vort_x'] = '(dy(w) - v_z)'
    problem.substitutions['Vort_y'] = '( u_z  - dx(w))'
    problem.substitutions['Vort_z'] = '(dx(v) - dy(u))'
    problem.substitutions['enstrophy']   = '(Vort_x**2 + Vort_y**2 + Vort_z**2)'

    # Mag defns
    problem.substitutions['Bx_rms'] = 'sqrt(Bx**2)'
    problem.substitutions['By_rms'] = 'sqrt(By**2)'
    problem.substitutions['Bz_rms'] = 'sqrt(Bz**2)'
    problem.substitutions['B_rms']  = 'sqrt(Bx**2 + By**2 + Bz**2)'
    problem.substitutions['B_perp'] = 'sqrt(Bx**2 + By**2)'

    #thermo
    problem.substitutions['s_over_cp'] = '((1/ɣ)*log(T_full) - ((ɣ-1)/ɣ)*log(rho_full))'
    problem.substitutions['P_full']    = 'R*rho_full*T_full'

    #Diffusivities
    problem.substitutions['nu']        = 'μ/rho_full'
    problem.substitutions['chi']       = 'K/(rho_full*Cp)'
    problem.substitutions['Re_rms']    = 'vel_rms*Lz/nu'
    problem.substitutions['Pe_rms']    = 'vel_rms*Lz/chi'
    problem.substitutions['Ma_rms']    = 'vel_rms/sqrt(R*T_full)'

    #Energies
    problem.substitutions['KE']        = 'rho_full*vel_rms**2/2'
    problem.substitutions['IE']        = 'rho_full*Cv*T_full'
    problem.substitutions['PE']        = 'rho_full*g*(z-1)'
    problem.substitutions['BE']        = 'B_rms/2'
    problem.substitutions['PE_fluc']   = '(PE - rho0*g*(z-1))'
    problem.substitutions['IE_fluc']   = '(IE - rho0*Cv*T0)'
    problem.substitutions['TE']        = '(KE + IE + PE + BE)'
    problem.substitutions['TE_fluc']   = '(KE + BE + IE_fluc + PE_fluc)'

    #Fluxes
    problem.substitutions['ohm_flux_z']   = 'μ0*η*(Jx*By - Jy*Bx)'
    problem.substitutions['poynt_flux_z'] = '(-(u*Bx + v*By)*Bz + (Bx**2 + By**2)*w)'
    problem.substitutions['enth_flux_z']  = '(w*(IE+P_full))'
    problem.substitutions['KE_flux_z']    = '(w*KE)'
    problem.substitutions['PE_flux_z']    = '(w*PE)'
    problem.substitutions['visc_flux_z']  = '(-μ * (u*Sig_xz + v*Sig_yz + w*Sig_zz))'
    problem.substitutions['F_cond_z']     = '(-K*dz(T_full))'
    problem.substitutions['F_cond0_z']    = '(-K*dz(T0))'
    problem.substitutions['F_cond1_z']    = '(-K*dz(T1))'

    return problem

def initialize_output(solver, domain, data_dir,
                      max_writes=10, max_vol_writes=2, output_dt=1, output_vol_dt=20,
                      mode="overwrite", volumes_output=True, coeff_output=False):
    """
    Sets up Dedalus output tasks for a Boussinesq convection run.

    Parameters
    ----------
    domain       : DedalusDomain object
        Contains information about the dedalus domain of the simulation
    de_problem      : DedalusProblem object
        Contains information aobut the dedalus problem & solver of the simulation
    data_dir        : string
        path to root data directory
    max_writes      : int, optional
        Maximum number of simulation output writes per file
    max_vol_writes  : int, optional
        Maximum number os imulations output writes per 3D volume file
    output_dt       : float, optional
        Simulation time between output writes
    output_vol_dt   : float, optional
        Simulation time between 3D volume output writes.
    mode            : string, optional
        Write mode for dedalus, "overwrite" or "append"
    volumes_output  : bool, optional
        If True, write 3D volumes
    coeff_output    : bool, optional
        If True, write coefficient data
    """

    analysis_tasks = analysis_tasks = OrderedDict()

    analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=max_writes, parallel=False, sim_dt=output_dt, mode=mode)
    analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", max_writes=max_writes, parallel=False, sim_dt=output_dt, mode=mode)

    basic_fields  = ['u_rms', 'v_rms', 'w_rms', 'vel_rms', 'enstrophy', 'T1', 'T1_z', 'T_full', 'ln_rho1', 'rho_full', 'Bx', 'By', 'Bz', 's_over_cp']
    fluid_numbers = ['Re_rms', 'Pe_rms', 'Ma_rms']
    energies      = ['KE', 'PE', 'IE', 'BE', 'TE', 'PE_fluc', 'IE_fluc', 'TE_fluc']
    fluxes        = ['ohm_flux_z', 'poynt_flux_z', 'enth_flux_z', 'KE_flux_z', 'PE_flux_z', 'visc_flux_z', 'F_cond_z', 'F_cond0_z', 'F_cond1_z']
    out_fields = basic_fields + fluid_numbers + energies + fluxes

    for field in out_fields:
        analysis_profile.add_task("plane_avg({})".format(field), name=field)
        analysis_scalar.add_task("vol_avg({})".format(field), name=field)
   
    analysis_profile.add_task("plane_avg(sqrt(T1**2))", name="T1_rms")
    analysis_scalar.add_task( "vol_avg(sqrt(T1**2))", name="T1_rms")
    analysis_scalar.add_task( "integ(  rho_full - rho0)", name="M1")

    analysis_tasks['profile'] = analysis_profile
    analysis_tasks['scalar'] = analysis_scalar

    ix, iy, iz = domain.bases[0].interval, domain.bases[1].interval, domain.bases[-1].interval
    slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=max_writes, mode=mode)
    for field in ['s_over_cp', 'enstrophy', 'u', 'w', 'T1', 'Vort_y', 'Vort_x']:
        slices.add_task("interp({},         y={})".format(field, (iy[0] + iy[1])/2),          name='{}'.format(field))
        slices.add_task("interp({},         z={})".format(field, iz[0] + (iz[1]-iz[0])*0.95), name='{} near top'.format(field))
        slices.add_task("interp({},         z={})".format(field, iz[0] + (iz[1]-iz[0])*0.05), name='{} near bot'.format(field))
        slices.add_task("interp({},         z={})".format(field, (iz[0] + iz[1])/2),          name='{} midplane'.format(field))
    analysis_tasks['slices'] = slices

    if volumes_output:
        analysis_volume = solver.evaluator.add_file_handler(data_dir+'volumes', sim_dt=output_vol_dt, max_writes=max_vol_writes, mode=mode)
        analysis_volume.add_task("T_full")
        analysis_volume.add_task("B_perp")
        analysis_volume.add_task("Bz")
        analysis_volume.add_task("u_perp")
        analysis_volume.add_task("w")
        analysis_tasks['volumes'] = analysis_volume

    return analysis_tasks

