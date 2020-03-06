import numpy as np
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict

def initialize_output(solver, domain, data_dir,
                      max_writes=10, max_vol_writes=2, output_dt=1, output_vol_dt=20,
                      mode="overwrite", volumes_output=True, coeff_output=False, magnetic=True, threeD=True):
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
    if not magnetic:
        bad_ks = ['Bx', 'By', 'Bz', 'BE', 'ohm_flux_z', 'poynt_flux_z']
        for k in bad_ks: out_fields.remove(k)
    if not threeD:
        out_fields.remove('v_rms')
        

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
    slice_fields = ['s_over_cp', 'enstrophy', 'u', 'w', 'T1', 'Vort_y', 'Vort_x', 'Bx', 'By', 'Bz']
    if not magnetic:
        bad_ks = ['Bx', 'By', 'Bz']
        for k in bad_ks: slice_fields.remove(k)
    if not threeD:
        slice_fields.remove('Vort_x')
    for field in slice_fields:
        if threeD:
            slices.add_task("interp({},         y={})".format(field, (iy[0] + iy[1])/2),          name='{}'.format(field))
            slices.add_task("interp({},         z={})".format(field, iz[0] + (iz[1]-iz[0])*0.95), name='{} near top'.format(field))
            slices.add_task("interp({},         z={})".format(field, iz[0] + (iz[1]-iz[0])*0.05), name='{} near bot'.format(field))
            slices.add_task("interp({},         z={})".format(field, (iz[0] + iz[1])/2),          name='{} midplane'.format(field))
        else:
            slices.add_task("{}".format(field), name='{}'.format(field))
    analysis_tasks['slices'] = slices

    if volumes_output and threeD:
        analysis_volume = solver.evaluator.add_file_handler(data_dir+'volumes', sim_dt=output_vol_dt, max_writes=max_vol_writes, mode=mode)
        analysis_volume.add_task("T_full")
        if magnetic:
            analysis_volume.add_task("B_perp")
            analysis_volume.add_task("Bz")
            analysis_volume.add_task("u_perp")
            analysis_volume.add_task("w")
        analysis_tasks['volumes'] = analysis_volume

    return analysis_tasks

