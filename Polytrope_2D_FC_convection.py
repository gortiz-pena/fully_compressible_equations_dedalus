"""
Dedalus script for fully compressible convection.

Usage:
    Polytrope_2D_FC_convection.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e2]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --epsilon=<eps>            Superadiabatic excess [default: 1e-4]
    --n_rho=<n_rho>            Number of density scale heights [default: 3]
    --gamma=<gamma>            Adiabatic index [default: 5/3]
    --aspect=<aspect>          Aspect ratio of problem [default: 4]
    --seed=<seed>              Random seed for initial noise [default: 42]
    --nz=<nz>                  Vertical resolution [default: 32]
    --nx=<nx>                  Horizontal resolution [default: 64]
    --ny=<nx>                  Horizontal resolution [default: 64]

    --FT                       If flagged, use FT boundary conditions (default is TT)
    --FF                       If flagged, use FF boundary conditions (default is TT)

    --mesh=<mesh>              Processor mesh if distributing 3D run in 2D 
    
    --run_time_wall=<time>     Run time, in hours [default: 23.5]
    --run_time_buoy=<time>     Run time, in buoyancy times [default: 500]
    --run_time_diff=<time_>    Run time, in diffusion times

    --restart=<file>           Restart from checkpoint file
    --overwrite                If flagged, force file mode to overwrite
    --seed=<seed>              RNG seed for initial conditoins [default: 42]

    --label=<label>            Optional additional case name label
    --root_dir=<dir>           Root directory for output [default: ./]
    --safety=<s>               CFL safety factor [default: 0.7]
    --RK222                    Use RK222 instead of RK443
"""
import logging
import os
import sys
import time
from fractions import Fraction

import numpy as np
from docopt import docopt
from mpi4py import MPI

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

from logic.output        import initialize_output
from logic.checkpointing import Checkpoint
from logic.fc_equations  import FCEquations2D
from logic.polytrope     import Polytrope
from logic.functions     import global_noise

logger = logging.getLogger(__name__)
args = docopt(__doc__)

### 1. Read in command-line args, set up data directory
data_dir = args['--root_dir'] + '/' + sys.argv[0].split('.py')[0]


data_dir += "_Ra{}_Pr{}_n_rho{}_eps{}_a{}".format(args['--Rayleigh'], args['--Prandtl'], args['--n_rho'], args['--epsilon'], args['--aspect'])
if args['--label'] is not None:
    data_dir += "_{}".format(args['--label'])
data_dir += '/'
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.makedirs('{:s}/'.format(data_dir))
    logdir = os.path.join(data_dir,'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
logger.info("saving run in: {}".format(data_dir))


run_time_buoy = args['--run_time_buoy']
run_time_diff = args['--run_time_diff']
run_time_wall = float(args['--run_time_wall'])
if run_time_diff is not None:
    run_time_diff = float(run_time_diff)
if run_time_buoy is not None:
    run_time_buoy = float(run_time_buoy)

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]

### 2. Setup Dedalus domain, problem, and substitutions/parameters
nx = int(args['--nx'])
ny = int(args['--ny'])
nz = int(args['--nz'])
aspect = float(args['--aspect'])

logger.info("Simulation resolution = {}x{}".format(nx, nz))

n_rho = float(args['--n_rho'])
epsilon = float(args['--epsilon'])
gamma   = float(Fraction(args['--gamma']))

atmosphere = Polytrope(n_rho, epsilon, gamma=gamma)
Lz = atmosphere.Lz
x_basis = de.Fourier(  'x', nx, interval = [0, Lz*aspect], dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval = [0, Lz],        dealias=3/2)

bases = [x_basis, z_basis]
domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)
z = domain.grid(-1)

equations = FCEquations2D()
problem = de.IVP(domain, variables=equations.variables, ncc_cutoff=1e-10)
atmosphere.build_atmosphere(domain, problem)

### 2. Simulation parameters
Ra = float(args['--Rayleigh'])
Pr = float(args['--Prandtl'])
t_buoy, t_diff = atmosphere.set_parameters(Ra=Ra, Pr=Pr, aspect=aspect)


### 4.Setup equations and Boundary Conditions
problem = equations.define_subs(problem)
for k, eqn in equations.equations.items():
    logger.info('Adding eqn "{:13s}" of form: "{:s}"'.format(k, eqn))
    problem.add_equation(eqn)

bcs = ['temp_L', 'temp_R', 'stressfree', 'impenetrable']
if args['--FT']:
    bcs.remove('temp_L')
    bcs.append('flux_L')
if args['--FF']:
    bcs.remove('temp_L')
    bcs.remove('temp_R')
    bcs.append('flux_L')
    bcs.append('flux_R')
for k, bc in equations.BCs.items():
    for bc_type in bcs:
        if bc_type in k:
            logger.info('Adding BC "{:15s}" of form: "{:s}" (condition: {})'.format(k, bc[0], bc[1]))
            problem.add_bc(bc[0], condition=bc[1])

### 5. Build solver
# Note: SBDF2 timestepper does not currently work with AE.
#ts = de.timesteppers.SBDF2
if args['--RK222']:
    ts = de.timesteppers.RK222
else:
    ts = de.timesteppers.RK443
cfl_safety = float(args['--safety'])
solver = problem.build_solver(ts)
logger.info('Solver built')

### 6. Set initial conditions: noise or loaded checkpoint
checkpoint = Checkpoint(data_dir)
checkpoint_dt = 25*t_buoy
restart = args['--restart']
not_corrected_times = True
if restart is None:
    noise = global_noise(domain)

    T0 = atmosphere.T0
    T0.set_scales(domain.dealias, keep_data=True)
    z_de = domain.grid(-1, scales=domain.dealias)

    T1 = solver.state['T1']
    T1_z = solver.state['T1_z']
    T1.set_scales(domain.dealias)
    T1['g'] = 1e-4*epsilon*T0['g']*np.sin(np.pi*z_de/Lz)*noise['g']
    T1.differentiate('z', out=T1_z)

    dt = None
    mode = 'overwrite'
else:
    logger.info("restarting from {}".format(restart))
    dt = checkpoint.restart(restart, solver)
    mode = 'append'
    not_corrected_times = False
checkpoint.set_checkpoint(solver, sim_dt=checkpoint_dt, mode=mode)
   

### 7. Set simulation stop parameters, output, and CFL
if run_time_diff is not None:   solver.stop_sim_time = run_time_diff*t_diff + solver.sim_time
elif run_time_buoy is not None: solver.stop_sim_time = run_time_buoy*t_buoy + solver.sim_time
else:                            solver.stop_sim_time = 1 + solver.sim_time
solver.stop_wall_time = run_time_wall*3600.

#TODO: Check max_dt, cfl, etc.
max_dt    = 0.1*t_buoy
if dt is None: dt = max_dt
analysis_tasks = initialize_output(solver, domain, data_dir, mode=mode, magnetic=False, threeD=False)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
CFL.add_velocities(('u', 'w'))


### 8. Setup flow tracking for terminal output, including rolling averages
#TODO: define these properly, probably only need Nu, KE, log string
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("Re_rms", name='Re')
flow.add_property("KE", name='KE')

Hermitian_cadence = 100
first_step = True
# Main loop
try:
    count = Re_avg = 0
    logger.info('Starting loop')
    init_time = last_time = solver.sim_time
    start_iter = solver.iteration
    start_time = time.time()
    avg_nu = avg_temp = avg_tz = 0
    while (solver.ok and np.isfinite(Re_avg)) or first_step:
        if first_step: first_step = False

        dt = CFL.compute_dt()
        solver.step(dt) #, trim=True)

        # Solve for blow-up over long timescales in 3D due to hermitian-ness
        effective_iter = solver.iteration - start_iter
        if effective_iter % Hermitian_cadence == 0:
            for field in solver.state.fields:
                field.require_grid_space()
        
                   
        if effective_iter % 1 == 0:
            Re_avg = flow.grid_average('Re')
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string += 'Time: {:8.3e} ({:8.3e} buoy / {:8.3e} diff), dt: {:8.3e}, '.format(solver.sim_time, solver.sim_time/t_buoy, solver.sim_time/t_diff,  dt)
            log_string += 'Re: {:8.3e}/{:8.3e}, '.format(Re_avg, flow.max('Re'))
            log_string += 'KE: {:8.3e}/{:8.3e}, '.format(flow.grid_average('KE'), flow.max('KE'))
            logger.info(log_string)
except:
    raise
    logger.error('Exception raised, triggering end of main loop.')
finally:
    end_time = time.time()
    main_loop_time = end_time-start_time
    n_iter_loop = solver.iteration-1
    logger.info('Iterations: {:d}'.format(n_iter_loop))
    logger.info('Sim end time: {:f}'.format(solver.sim_time))
    logger.info('Run time: {:f} sec'.format(main_loop_time))
    logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
    logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
    try:
        final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
        final_checkpoint.set_checkpoint(solver, wall_dt=1, mode=mode)
        solver.step(dt) #clean this up in the future...works for now.
        post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
    except:
        raise
        print('cannot save final checkpoint')
    finally:
        logger.info('beginning join operation')
        post.merge_analysis(data_dir+'checkpoint')

        for key, task in analysis_tasks.items():
            logger.info(task.base_path)
            post.merge_analysis(task.base_path)

        logger.info(40*"=")
        logger.info('Iterations: {:d}'.format(n_iter_loop))
        logger.info('Sim end time: {:f}'.format(solver.sim_time))
        logger.info('Run time: {:f} sec'.format(main_loop_time))
        logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
        logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
