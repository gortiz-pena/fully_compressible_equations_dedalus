"""
Script for plotting traces of evaluated scalar quantities vs. time.

Usage:
    plot_scalars.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Output directory for figures [default: traces]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
"""
import logging
logger = logging.getLogger(__name__)
from docopt import docopt
args = docopt(__doc__)
from plot_logic.scalars import ScalarFigure, ScalarPlotter

root_dir = args['<root_dir>']
fig_name  = args['--fig_name']
start_file = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)


# Ma vs time
fig1 = ScalarFigure(1, 1, col_in=6, fig_name='ma_trace')
fig1.add_field(0, 'Ma_rms')

# Nu vs time
figNu = ScalarFigure(1, 1, col_in=6, fig_name='nu_trace')
figNu.add_field(0, 'Nu')


# Re vs. time
fig2 = ScalarFigure(1, 1, col_in=6, fig_name='pe_trace')
fig2.add_field(0, 'Pe_rms')

# dT 
fig3 = ScalarFigure(1, 1, col_in=6, fig_name='s_over_cp_z')
fig3.add_field(0, 's_over_cp_z')

# Energies
fig4 = ScalarFigure(5, 1, col_in=8, row_in=2.5, fig_name='energies')
fig4.add_field(0, 'KE')
fig4.add_field(0, 'IE_fluc')
fig4.add_field(0, 'PE_fluc')
fig4.add_field(0, 'TE_fluc')
fig4.add_field(1, 'KE')
fig4.add_field(2, 'IE_fluc')
fig4.add_field(3, 'PE_fluc')
fig4.add_field(4, 'TE_fluc')

# Load in figures and make plots
plotter = ScalarPlotter(root_dir, file_dir='scalar', fig_name=fig_name, start_file=start_file, n_files=n_files)
figs = [fig1, fig2, fig3, fig4, figNu]
plotter.load_figures(figs)
plotter.plot_figures(dpi=int(args['--dpi']))
plotter.plot_convergence_figures(dpi=int(args['--dpi']))
