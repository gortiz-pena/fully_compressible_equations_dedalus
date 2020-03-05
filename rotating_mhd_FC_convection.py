"""
Dedalus script for fully compressible convection.

Usage:
    rotating_mhd_FC_convection.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 3e7]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --Taylor=<Taylor>          Taylor number [default: 5e8]
    --Pm=<Pm>                  Mag. Prandtl number [default: 1]
    --nz=<nz>                  Vertical resolution [default: 256]
    --nx=<nx>                  Horizontal resolution [default: 256]
    --ny=<nx>                  Horizontal resolution [default: 256]
    --aspect=<aspect>          Aspect ratio of problem [default: 2]

    --mesh=<mesh>              Processor mesh if distributing 3D run in 2D 
    
    --run_time_wall=<time>     Run time, in hours [default: 23.5]
    --run_time_buoy=<time>     Run time, in buoyancy times
    --run_time_diff=<time_>    Run time, in diffusion times [default: 1]

    --restart=<file>           Restart from checkpoint file
    --overwrite                If flagged, force file mode to overwrite
    --seed=<seed>              RNG seed for initial conditoins [default: 42]

    --label=<label>            Optional additional case name label
    --root_dir=<dir>           Root directory for output [default: ./]
    --safety=<s>               CFL safety factor [default: 0.4]
    --RK443                    Use RK443 instead of RK222
"""
import logging
import os
import sys
import time

import numpy as np
from docopt import docopt
from mpi4py import MPI

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

from logic.output import define_output_subs, initialize_output
from logic.checkpointing import Checkpoint

logger = logging.getLogger(__name__)
args = docopt(__doc__)

### 1. Read in command-line args, set up data directory
data_dir = args['--root_dir'] + '/' + sys.argv[0].split('.py')[0]


data_dir += "_Ra{}_Ta{}_Pr{}_Pm{}_a{}".format(args['--Rayleigh'], args['--Taylor'], args['--Prandtl'], args['--Pm'], args['--aspect'])
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
if run_time_buoy is not None:
    run_time_buoy = float(run_time_buoy)
if run_time_diff is not None:
    run_time_diff = float(run_time_diff)

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]




### 3. Setup Dedalus domain, problem, and substitutions/parameters
nx = int(args['--nx'])
ny = int(args['--ny'])
nz = int(args['--nz'])
aspect = float(args['--aspect'])

logger.info("Simulation resolution = {}x{}x{}".format(nx, ny, nz))

x_basis = de.Fourier(  'x', nx, interval = [0, aspect], dealias=3/2)
y_basis = de.Fourier(  'y', ny, interval = [0, aspect], dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval = [0, 1],      dealias=3/2)

bases = [x_basis, y_basis, z_basis]
domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)

variables = ['T1', 'T1_z', 'ln_rho1', 'u', 'v', 'w', 'u_z', 'v_z', 'w_z', 'Bx', 'By', 'Bz', 'Ax', 'Ay', 'Az', 'phi']
problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-10)

problem.parameters['Lx'] = problem.parameters['Ly'] = aspect
problem.parameters['Lz'] = d = 1

# Nondimensionalization
problem.parameters['g']  = g  = 1
problem.parameters['μ0'] = μ0 = 1
problem.parameters['Cp'] = Cp = 1
problem.parameters['Cv'] = Cv = 3/5
problem.parameters['ɣ']  = ɣ  = 5/3

### 2. Simulation parameters
Ra = float(args['--Rayleigh'])
Ta = float(args['--Taylor'])
Pr = float(args['--Prandtl'])
Pm = float(args['--Pm'])

logger.info("resolution = {}x{}x{}".format(nx, ny, nz))

#ds_dz_m = (1/gamma) * grad T_m / T_m  - (1-gamma)/gamma * grad rho_m / rho_m  
#...with grad T_m = -5/4, grad rho_m = -6/5, T_m = 25/24, rho_m = 1
rho_m = 1
ds_dz_m = (1/ɣ)*(-5/4) / (25/24) - (1-ɣ)/ɣ * (-6/5) / rho_m
K  = np.sqrt(g*d**4*rho_m**2 * Cp * -1 * ds_dz_m / Ra / Pr)
μ  = K * Pr / Cp
η  = μ * Pm
Ω0 = (μ / 4 / d**4) * np.sqrt(Ta)

problem.parameters['Ω0'] = Ω0 
problem.parameters['K']  = K 
problem.parameters['μ']  = μ
problem.parameters['η']  = η 
problem.substitutions['R']  = '(ɣ-1)*Cv'

### 2. Simulation parameters
Ra = float(args['--Rayleigh'])
Ta = float(args['--Taylor'])
Pr = float(args['--Prandtl'])
Pm = float(args['--Pm'])

logger.info("Ra = {:.3e}, Ta = {:.3e}, Pr = {:2g}, Pm = {:2g}".format(Ra, Ta, Pr, Pm))
logger.info("Ω0 = {:.3e}, K = {:.3e}, μ = {:2e}, η = {:2e}".format(Ω0, K, μ, η))


# Atmosphere
problem.substitutions['T0']               = '(5/12)*(4-3*z)'
problem.substitutions['T0_z']             = '(-5/4)'
problem.substitutions['rho0']             = '(2/5)*(4-3*z)'
problem.substitutions['ln_rho0']          = '(log(2/5) + log(4-3*z))'
problem.substitutions['ln_rho0_z']        = '(-3/(4-3*z))'
problem.substitutions['rho_full']         = '(rho0*exp(ln_rho1))'
problem.substitutions['T_full']           = '(T0 + T1)'

# Operators
problem.substitutions['Lap(A, A_z)']       = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
problem.substitutions['UdotGrad(A, A_z)']  = '(u*dx(A) + v*dy(A) + w*A_z)'
problem.substitutions['Div(Ax, Ay, Az_z)'] = '(dx(Ax) + dy(Ay) + Az_z)'
problem.substitutions['DivU']              = 'Div(u, v, w_z)'
problem.substitutions['plane_avg(A)']      = 'integ(A, "x", "y")/Lx/Ly'
problem.substitutions['vol_avg(A)']        = 'integ(A)/Lx/Ly/Lz'
problem.substitutions['plane_std(A)']      = 'sqrt(plane_avg((A - plane_avg(A))**2))'

#problem.substitutions['Bz'] = '(dx(Ay) - dy(Ax))'
problem.substitutions['Jx'] = '(dy(Bz) - dz(By))'
problem.substitutions['Jy'] = '(dz(Bx) - dx(Bz))'
problem.substitutions['Jz'] = '(dx(By) - dy(Bx))'

# Stress Tensor and other hard terms
problem.substitutions["Sig_xx"] = "(2*dx(u) - 2/3*DivU)"
problem.substitutions["Sig_yy"] = "(2*dy(v) - 2/3*DivU)"
problem.substitutions["Sig_zz"] = "(2*w_z   - 2/3*DivU)"
problem.substitutions["Sig_xy"] = "(dx(v) + dy(u))"
problem.substitutions["Sig_xz"] = "(dx(w) +  u_z )"
problem.substitutions["Sig_yz"] = "(dy(w) +  v_z )"

problem.substitutions['visc_u']   = "( Lap(u, u_z) + 1/3*dx(DivU) )"
problem.substitutions['visc_v']   = "( Lap(v, v_z) + 1/3*dy(DivU) )"
problem.substitutions['visc_w']   = "( Lap(w, w_z) + 1/3*Div(u_z, v_z, dz(w_z)) )"                

problem.substitutions['visc_u_L'] = 'μ*visc_u*(2/rho0)'
problem.substitutions['visc_v_L'] = 'μ*visc_v*(2/rho0)'
problem.substitutions['visc_w_L'] = 'μ*visc_w*(2/rho0)'
problem.substitutions['visc_u_R'] = 'μ*visc_u*(1/rho_full - 2/rho0)'
problem.substitutions['visc_v_R'] = 'μ*visc_v*(1/rho_full - 2/rho0)'
problem.substitutions['visc_w_R'] = 'μ*visc_w*(1/rho_full - 2/rho0)'

problem.substitutions['visc_heat']  = " μ*(dx(u)*Sig_xx + dy(v)*Sig_yy + w_z*Sig_zz + Sig_xy**2 + Sig_xz**2 + Sig_yz**2)"
problem.substitutions['ohm_heat']   = 'μ0*η*(Jx**2 + Jy**2 + Jz**2)'

problem = define_output_subs(problem)

### 4.Setup equations and Boundary Conditions
problem.add_equation("Bx + dz(Ay) - dy(Az) = 0") #
problem.add_equation("By + dx(Az) - dz(Ax) = 0") #
problem.add_equation("Bz + dy(Ax) - dx(Ay) = 0")
problem.add_equation("dz(T1) - T1_z = 0") #
problem.add_equation("dz(u)  - u_z  = 0") #
problem.add_equation("dz(v)  - v_z  = 0") #
problem.add_equation("dz(w)  - w_z  = 0") #
problem.add_equation("T0*(dt(ln_rho1) + w*ln_rho0_z + DivU) = -T0*UdotGrad(ln_rho1, dz(ln_rho1))")
problem.add_equation("dt(Ax) + μ0*η*Jx + dx(phi) = v*Bz - w*By") #
problem.add_equation("dt(Ay) + μ0*η*Jy + dy(phi) = w*Bx - u*Bz") #
problem.add_equation("dt(Az) + μ0*η*Jz + dz(phi) = u*By - v*Bx") #
problem.add_equation("T0*(dt(u)  + R*(dx(T1) + T0*dx(ln_rho1))                - 2*Ω0*v - visc_u_L) = T0*(-UdotGrad(u, u_z) - R*T1*dx(ln_rho1) + (1/rho_full)*(Jy*Bz - Jz*By) + visc_u_R)") #
problem.add_equation("T0*(dt(v)  + R*(dy(T1) + T0*dy(ln_rho1))                + 2*Ω0*u - visc_v_L) = T0*(-UdotGrad(v, v_z) - R*T1*dy(ln_rho1) + (1/rho_full)*(Jz*Bx - Jx*Bz) + visc_v_R)") #
problem.add_equation("T0*(dt(w)  + R*(T1_z   + T0*dz(ln_rho1) + T1*ln_rho0_z)          - visc_w_L) = T0*(-UdotGrad(w, w_z) - R*T1*dz(ln_rho1) + (1/rho_full)*(Jx*By - Jy*Bx) + visc_w_R)") ##
problem.add_equation("T0*(dt(T1) + w*T0_z + (ɣ-1)*T0*DivU - 2*(ɣ*K)*Lap(T1, T1_z)/rho0           ) = T0*(-UdotGrad(T1, T1_z) - (ɣ-1)*T1*DivU + (ɣ*K)*Lap(T1, T1_z)*(1/rho_full - 2/rho0) + visc_heat + ohm_heat)") #
problem.add_equation("Div(Ax, Ay, dz(Az)) = 0") #

logger.info("Thermal BC: fixed temperature")
problem.add_bc(" left(T1) = 0")
problem.add_bc("right(T1)  = 0")

logger.info("Magnetic BC: Horizontal components set to zero")
problem.add_bc(" left(Bx) = 0")
problem.add_bc("right(Bx) = 0")
problem.add_bc(" left(By) = 0")
problem.add_bc("right(By) = 0")

logger.info("Horizontal velocity BC: stress free")
problem.add_bc(" left(u_z) = 0")
problem.add_bc("right(u_z) = 0")
problem.add_bc(" left(v_z) = 0")
problem.add_bc("right(v_z) = 0")

logger.info("Vertical velocity BC: impenetrable")
problem.add_bc( "left(w) = 0")
problem.add_bc("right(w) = 0")

problem.add_bc(" left(phi) = 0")
problem.add_bc("right(phi) = 0")

### 5. Build solver
# Note: SBDF2 timestepper does not currently work with AE.
#ts = de.timesteppers.SBDF2
if args['--RK443']:
    ts = de.timesteppers.RK443
else:
    ts = de.timesteppers.RK222
cfl_safety = float(args['--safety'])
solver = problem.build_solver(ts)
logger.info('Solver built')

delta_S = np.abs(np.log(1/4))
t_buoy  = np.sqrt(g*Cp*d/delta_S) 
t_diff  = np.sqrt(d/μ)


### 6. Set initial conditions: noise or loaded checkpoint
checkpoint = Checkpoint(data_dir)
checkpoint_dt = 100
restart = args['--restart']
not_corrected_times = True
if restart is None:
    x_de = domain.grid(0, scales=domain.dealias)
    y_de = domain.grid(1, scales=domain.dealias)
    z_de = domain.grid(-1, scales=domain.dealias)

    f1 = 1.0*np.sin(34*np.pi*x_de/aspect) + 1.1*np.cos(32*np.pi*x_de/aspect) + 0.9*np.sin(38*np.pi*x_de/aspect)
    f2 = 1.0*np.cos(24*np.pi*x_de/aspect) - 0.7*np.sin(28*np.pi*x_de/aspect) + 1.4*np.cos(34*np.pi*x_de/aspect)
    g1 = 0.6*np.cos(38*np.pi*y_de/aspect) - 1.0*np.sin(46*np.pi*y_de/aspect) - 1.2*np.sin(28*np.pi*y_de/aspect)
    g2 = 1.0*np.cos(32*np.pi*y_de/aspect) + 0.8*np.sin(36*np.pi*y_de/aspect) - 1.1*np.cos(42*np.pi*y_de/aspect)
    h1 = 1.0*np.sin(13*np.pi*z_de) - 0.8*np.sin(16*np.pi*z_de) + 1.2*np.sin(20*np.pi*z_de)
    h2 = 1.0*np.sin(15*np.pi*z_de) - 1.5*np.sin(12*np.pi*z_de) + 0.7*np.sin(18*np.pi*z_de)

    T1 = solver.state['T1']
    T1_z = solver.state['T1_z']
    T1.set_scales(domain.dealias)
    T1['g'] = 2e-3*(f1*g1*h1 + f2*g2*g2)
    T1.differentiate('z', out=T1_z)

    Bz   = solver.state['Bz']
    Bz.set_scales(domain.dealias)
    Bz['g'] = -3e-6*np.cos(20*np.pi*x_de/aspect)*np.cos(20*np.pi*y_de/aspect)

    dt = None
    mode = 'overwrite'
else:
    logger.info("restarting from {}".format(restart))
    dt = checkpoint.restart(restart, solver)
    mode = 'append'
    not_corrected_times = False
checkpoint.set_checkpoint(solver, sim_dt=checkpoint_dt, mode=mode)
   

### 7. Set simulation stop parameters, output, and CFL
if run_time_buoy is not None:    solver.stop_sim_time = run_time_buoy*t_buoy + solver.sim_time
elif run_time_diff is not None: solver.stop_sim_time = run_time_diff*t_diff + solver.sim_time
else:                            solver.stop_sim_time = 1 + solver.sim_time
solver.stop_wall_time = run_time_wall*3600.

#TODO: Check max_dt, cfl, etc.
max_dt    = np.min((1e-1, t_diff, t_buoy))
if dt is None: dt = max_dt
analysis_tasks = initialize_output(solver, domain, data_dir, mode=mode)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
CFL.add_velocities(('u', 'v', 'w'))


### 8. Setup flow tracking for terminal output, including rolling averages
#TODO: define these properly, probably only need Nu, KE, log string
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("Re_rms", name='Re')
flow.add_property("KE", name='KE')
flow.add_property("B_rms", name='B_rms')
flow.add_property("Div(Bx, By, dz(Bz))", name='DivB')

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
            log_string += 'B:  {:8.3e}/{:8.3e}, '.format(flow.grid_average('B_rms'), flow.max('B_rms'))
            log_string += 'divB:  {:8.3e}/{:8.3e}'.format(flow.grid_average('DivB'), flow.max('DivB'))
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
