from collections import OrderedDict
import glob
from copy import deepcopy

import h5py
import numpy as np
from mpi4py import MPI
import dedalus.public as de


def sort_file_list(f_list):
    """
    Given a list of .h5 files, this function sorts them and returns a list of sorted files.
    """

    files = []
    for f in f_list:
        file_num = int(f.split('.h5')[0].split('_s')[-1])
        files.append((f, file_num))
    sorted_list, numbers = zip(*sorted(files, key=lambda x: x[1]))
    return sorted_list


def load_tt(path, time, Lz):
    """
    Loads in data from a TT system

    Inputs:
    -------
    See tt_to_ft() docstring.

    Lz : float
         Depth of dedalus domain
    
    """
    tt_dirs = glob.glob('{:s}/*/'.format(path))
    sub_dirs = [d.split('/')[-2] for d in tt_dirs]
    if 'final_checkpoint' in sub_dirs:
        final_checks = sort_file_list(glob.glob('{:s}/final_checkpoint/*.h5'.format(path)))
        checkpoint_TT = final_checks[-1]
    else:
        checks = sort_file_list(glob.glob('{:s}/checkpoint/*.h5'.format(path)))
        checkpoint_TT = checks[-1]

    # Figure out how many scalar / profile files are needed
    scalar_files  = sort_file_list(glob.glob(  '{:s}/scalar/*.h5'.format(path)))
    profile_files = sort_file_list(glob.glob('{:s}/profiles/*.h5'.format(path)))
    end_times = [None, None]

    good_scalar_files = []
    good_profile_files = []
    n_scalar_writes = 0
    n_profile_writes = 0
    for i in range(len(scalar_files)):
        this_file = scalar_files[-(i+1)]
        good_scalar_files.append(this_file)
        with h5py.File(this_file, 'r') as sf:
            sim_times = sf['scales']['sim_time'][()]
            n_scalar_writes += len(sim_times)
            if end_times[0] is None:
                end_times[0] = sim_times[-1]
            if end_times[0] - sim_times[0] > time: 
                break
    for i in range(len(profile_files)):
        this_file = profile_files[-(i+1)]
        good_profile_files.append(this_file)
        with h5py.File(this_file, 'r') as pf:
            sim_times = pf['scales']['sim_time'][()]
            nz = len(pf['scales/z/1.0'][()])
            n_profile_writes += len(sim_times)
            if end_times[1] is None:
                end_times[1] = sim_times[-1]
            if end_times[1] - sim_times[0] > time: 
                break

    good_scalar_files  = sort_file_list(good_scalar_files)
    good_profile_files = sort_file_list(good_profile_files)

    #Get average profiles / scalars
    profiles = OrderedDict()
    scalars = OrderedDict()
    scalar_work_field = np.zeros(n_scalar_writes)
    profile_work_field = np.zeros((n_scalar_writes, nz))
    for sk in ['Nu', 's_over_cp_z']:
        i = 0
        for this_file in good_scalar_files:
            with h5py.File(this_file, 'r') as sf:
                N = len(sf['scales']['sim_time'][()])
                scalar_work_field[i:i+N] = sf['tasks'][sk][()].squeeze()
                i += N
        scalars[sk] = np.mean(scalar_work_field)

    for pk in ['UdotGradw', 'T1']:
        i = 0
        for this_file in good_profile_files:
            with h5py.File(this_file, 'r') as pf:
                N = len(pf['scales']['sim_time'][()])
                profile_work_field[i:i+N,:] = pf['tasks'][pk][()].squeeze()
                i += N
        profiles[pk] = np.mean(profile_work_field, axis=0)
    
    scalars['s_over_cp_z'] *= Lz # take it from vol_avg to integral quantity
    return checkpoint_TT, profiles, scalars

def structure_bvp(atmo_class, atmo_args, atmo_kwargs, profiles, scalars):
    """
    Solves a BVP to get the right FT atmospheric structure

    Inputs:
    -------
        atmo_class : The class type of the atmosphere object
        atmo_args : tuple
            args for atmo_class.__init__()
        atmo_kwargs : dict
            kwargs for atmo_class.__init__()
        profiles : OrderedDict
            A dictionary with profiles required for the BVP
        scalars : OrderedDict
            A dictionary with scalar values required for the BVP
    """

    dS_TT = scalars['s_over_cp_z']
    Nu    = scalars['Nu']
    T1_TT = profiles['T1']
    UdotGradw_TT = profiles['UdotGradw']

    atmosphere = atmo_class(*atmo_args, **atmo_kwargs)
    Lz = atmosphere.Lz

    z_basis = de.Chebyshev('z', len(T1_TT), interval=[0, Lz], dealias=1)
    domain  = de.Domain([z_basis,], grid_dtype=np.float64, comm=MPI.COMM_SELF)
    problem = de.NLBVP(domain, variables = ['dS', 'S', 'ln_rho1', 'M1'])
    atmosphere.build_atmosphere(domain, problem)
    dS_TT *= atmosphere.Cp


    z = domain.grid(0)

    #Solve out for Temp profile
    S0 = domain.new_field()
    T_TT = domain.new_field()
    T1_FT = domain.new_field()
    UdotGradw = domain.new_field()
    grad_ad = -(atmosphere.g/atmosphere.Cp)

    T_TT['g'] = atmosphere.T0['g'] + T1_TT
    T_ad = 1 + grad_ad*(z - Lz)
    dT_ad_TT = np.mean(T_TT.interpolate(z=Lz)['g'] - T_TT.interpolate(z=Lz)['g']) - np.abs(grad_ad*Lz)
    dT_ad_FT = dT_ad_TT / Nu
    T1_FT['g'] =  dT_ad_FT*((T_TT['g'] - T_ad)/dT_ad_TT) + T_ad - atmosphere.T0['g']

    #Get dS0
    S0['g'] = atmosphere.Cp*((1/atmosphere.ɣ)*np.log(atmosphere.T0['g']) - ((atmosphere.ɣ-1)/atmosphere.ɣ)*np.log(atmosphere.rho0['g']))
    dS0 = np.mean(S0.interpolate(z=Lz)['g'] - S0.interpolate(z=0)['g'])
    evolved_Ra_factor = (dS_TT / dS0)

    UdotGradw['g'] = UdotGradw_TT

    problem.parameters['T1_FT'] = T1_FT
    problem.parameters['dS_TT'] = dS_TT
    problem.parameters['UdotGradw_TT'] = UdotGradw
    problem.substitutions['ln_rho0'] = 'log(rho0)'
    problem.add_equation("(T0 + T1_FT)*dz(ln_rho1) = -dz(T1_FT) - T1_FT*ln_rho0_z + (dS/dS_TT)*UdotGradw_TT")
    problem.add_equation("dz(M1) = rho0*(exp(ln_rho1) - 1)")
    problem.add_equation("S/Cp = (1/ɣ) * log(T0 + T1_FT) - ((ɣ-1)/ɣ)*(ln_rho0 + ln_rho1)")
    problem.add_equation("dS = right(S) - left(S)")

    problem.add_bc(" left(M1) = 0")
    problem.add_bc("right(M1) = 0")

    tol = 1e-10
    solver = problem.build_solver()
    dS = solver.state['dS']
    ln_rho1_FT = solver.state['ln_rho1']
    pert = solver.perturbations.data
    pert.fill(1+tol)
    while np.sum(np.abs(pert)) > tol:
        solver.newton_iteration()
        print('pert norm: {} / dS: {:.2e}'.format(np.sum(np.abs(pert)), np.mean(dS['g'])))

    dS_FT = np.mean(dS['g'])

    FT_Ra_factor = (dS0 / dS_FT)*evolved_Ra_factor
    
    return T1_FT['g'], ln_rho1_FT['g'], dS_FT/dS_TT, dT_ad_FT/dT_ad_TT, FT_Ra_factor

    


def tt_to_ft_preliminaries(atmosphere, atmo_args, atmo_kwargs, path, time):
    """
    Prepares to starts an FT simulation from an equilibrated TT simulation.

    Inputs:
    -------
        atmosphere : An atmosphere class object (e.g., Polytrope)
            The atmosphere containing information regarding the initial conditions of the run.
        atmo_args : tuple
            args for atmo_class.__init__()
        atmo_kwargs : dict
            kwargs for atmo_class.__init__()
        path : string
            A string path to the TT simulation to get the state of
        time : float
            The amount of simulation time to average profiles and scalar values over from the TT run
    """
    if MPI.COMM_WORLD.rank == 0:
        # Get average profiles/scalars from TT system
        checkpoint_TT, profiles, scalars = load_tt(path, time, atmosphere.Lz)

        # Solve TT-to-FT BVP
        T1_FT, ln_rho1_FT, dS_factor, dT_ad_factor, FT_Ra_factor = structure_bvp(atmosphere.__class__, atmo_args, atmo_kwargs, profiles, scalars)

    checkpoint_TT   = MPI.COMM_WORLD.bcast(checkpoint_TT, root=0)
    T1_FT           = MPI.COMM_WORLD.bcast(T1_FT, root=0)
    ln_rho1_FT      = MPI.COMM_WORLD.bcast(ln_rho1_FT, root=0)
    dS_factor       = MPI.COMM_WORLD.bcast(dS_factor, root=0)
    dT_ad_factor    = MPI.COMM_WORLD.bcast(dT_ad_factor, root=0)
    FT_Ra_factor    = MPI.COMM_WORLD.bcast(FT_Ra_factor, root=0)

    return checkpoint_TT, T1_FT, ln_rho1_FT, dS_factor, dT_ad_factor, FT_Ra_factor

def tt_to_ft(solver, checkpoint, checkpoint_TT, T1_FT, ln_rho1_FT, dS_factor, dT_ad_factor):
    """
    Starts an FT simulation from an equilibrated TT simulation.

    Inputs:
    -------
        solver : Dedalus solver object
            The solver for the FT system
        checkpoint : A Checkpoint object 
            A checkpoint object for loading states for the FT system
        checkpoint_TT : string
            path to TT checkpoint file
        T1_FT : NumPy array
            mean profile for T1 field in FT simulation
        ln_rho1_FT : NumPy array
            mean profile for ln_rho1 field in FT simulation
        dS_factor : float
            evolved entropy jump across domain in FT simulation (normalized by the TT one)
        dT_ad_factor : float
            evolved superaiabatic temperature jump across domain in FT simulation (normalized by the TT one)
    """
    dt = checkpoint.restart(checkpoint_TT, solver)
    T1 = solver.state['T1']
    T1_z = solver.state['T1_z']
    ln_rho1 = solver.state['ln_rho1']
    u  = solver.state['u']
    w  = solver.state['w']
    uz = solver.state['u_z']
    wz = solver.state['w_z']

    z_slice = solver.domain.dist.grid_layout.slices(scales=1)[-1]

    #Needs to be generalized to 3D

    T1.set_scales(solver.domain.dealias, keep_data=True)
    T1['g'] -= T1.integrate('x')['g']/solver.domain.bases[0].interval[-1]
    T1['g'] *= dT_ad_factor
    T1.set_scales(1, keep_data=True)
    T1['g'] += T1_FT[z_slice]
    T1.differentiate('z', out=T1_z)

    ln_rho1.set_scales(solver.domain.dealias, keep_data=True)
    ln_rho1['g'] -= ln_rho1.integrate('x')['g']/solver.domain.bases[0].interval[-1]
    ln_rho1['g'] *= dT_ad_factor
    ln_rho1.set_scales(1, keep_data=True)
    ln_rho1['g'] += ln_rho1_FT[z_slice]

    print(dT_ad_factor, dS_factor)


    u['g'] *= np.sqrt(dS_factor)
    w['g'] *= np.sqrt(dS_factor)
    u.differentiate('z', out=uz)
    w.differentiate('z', out=wz)

    dt /= np.sqrt(dS_factor)
    return solver, dt



