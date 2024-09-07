#!/usr/bin/env python3

import os
import re
import sys

#########################################################
#  Please modify this path to your working directory
#########################################################

workingDirectory = '/home/cristopher/GSI/btf-simulation/'

sys.path.insert(1, workingDirectory)

from src.btfsimulation import *

class Options(object):
    pass


np.random.seed(43543557)


DEBUG = __name__ != '__main__'
DEBUG = False

if DEBUG:
    # Import for testing and metadata usage
    print('! ! !   Simulation loaded in debugging mode   ! ! !\n')

###############################################################################

filename = 'parameters.json'
with open(filename, 'r') as json_file:
    params = json.load(json_file)

currentTime = datetime.datetime.now()
initTime    = currentTime

description = f"""
Particle tracking simulation of BTF experiment at HIT

Date: {currentTime}
"""

sim = BTFSimulation(description=description, slim=DEBUG, useCPU=True)

ex = os.path.basename(__file__)[4:-3]  # filename like run_***.py

output = f"output"
os.makedirs(output, exist_ok=True)

## Lattice
###############################################################################

path2Lattice = workingDirectory
sim.load_lattice(path2Lattice+'hitring_202312.json', septum_aperture_name='septum_begin')
sim.set_closed_orbit_correctors()

## Particles
###############################################################################

# number of particles in simulation
nparticles = params['nparticles']

sim.set_particle(
    ion=Ion('C12 6+', q='6 e', m='12 u', Ekin_per_u='125.24 MeV/u'),
    nparticles = nparticles,
)


## Excitation
###############################################################################

turns_init  = params['turns_init'] 
turnsPerExc = params['turnsPerExc']

excSamples = params['excSamples']

# Frequency samples of the external excitation
BW = 0.026  # bandwidth in MHz
exc_harmonic = params['band']

sgn = params['band_sgn']
 
fbet0 = (exc_harmonic + sgn * 0.678)
fbet  = np.linspace(fbet0-BW/2, fbet0+BW/2, excSamples)

if exc_harmonic == 8:
    print()
    print('Exc harmonic 8')
    print('Setting excitation span to measurement')
    print()
    finit = 15.865971/2.1721
    fend  = 15.92873/2.1721

    fbet = np.linspace(finit, fend, excSamples)

qSample = fbet

# Direction of sweep 
#qSample = qSample[::-1]

nturns = int(turns_init + turnsPerExc*excSamples) 

print('\nSimulation settings\n')
print(f'  N particles:           {nparticles} ({nparticles:.1e}) stored particles')
print(f'  Excitation steps:      {excSamples} ') # 701 in experiment
print(f'  Turns per excitation:  {turnsPerExc}') # 31k in experiment
print(f'  Turns (total):         {nturns} ({nturns:.1e})') # 22M in experiment


from src.gr_btf_signal_time import gr_btf_signal_time

excAmp  = params['excAmp']
band = params['band']

samples_per_turn = params['sg_samp_per_turn']
sampling_freq = samples_per_turn*sim.frev
print()
print(f'\nsampling_freq: {sampling_freq*1e-6:g} MHz')
print(f'rev_freq: {sim.frev*1e-6:g} MHz\n')

sg_samples = gr_btf_signal_time(qSample,excAmp=1.,
                                nPerSample=turnsPerExc,
                                f_rev=sim.frev,
                                sampling_freq=sampling_freq)

## Install exciter beam element
####################################

# HIT KO Exciter form and RF factor
# kick in rad per volt between plates (field calculation)
# k0l_per_U = 0.005567e-6 #+ 0.007519e-6 # E+B for 254.24 MeV/u C12 6+ 
# nominal voltage between plates
# k0l = k0l_per_U * U_ref

sim.install_exciter(
    exciter_name='s1bo1',  # name as in loaded lattice
    samples=sg_samples,
    start_turn=turns_init,
    k0l=excAmp,
    sampling=sampling_freq,
)

## Monitors
###############################################################################
# Note: the following error messages can indicate insufficient memory. Reduce buffers then!
#  - TypeError: _enqueue_write_buffer(): incompatible function arguments. Invoked with: [...] None
#  - Killed


print()
print(f"Revolution frequency set is {sim.frev*1e-6:g} MHz",)
print()
#schottky_pickup = 'hitring$start'
schottky_pickup = 's2dx1s'

sim.install_BeamPositionMonitor(
    name='BPM-Schottky',
    num_particles=nparticles,
    start_at_turn=0,
    stop_at_turn=nturns,
    rev_frequency=sim.frev,
    sampling_frequency=sampling_freq,
    at_element=schottky_pickup,
)

## Sextupole strength setup
###############################################################################

sim.complete_line()

kPrimeL = params['kPrimeL']

correctChroma = params['correctChroma']
#kPrimeL = 0.00

sim.set_sextupole_str(kPrimeL, correct_chroma=correctChroma)
sim.determine_sextupole()

sim.finish_setup()

## Complete setup and initialize beam
###############################################################################
dpp = params['dpp']
beamEmittance = params['beamEmittance']

sim.create_beam(
    emitt_x = Qty(f'{beamEmittance} mm mrad'),
    emitt_y = Qty(f'{beamEmittance} mm mrad'),
    rel_momentum_spread = Qty(f'{dpp}'), # relative momentum spread ( P/p0 - 1 )
    save=output,
)

## Run tracking
###############################################################################

if not DEBUG:
    # Actual simulation run
    
    # expected time on single GPU:
    # 14s for (2000 turns * 1e4 particles) = 7e-7 s/particle/turn
    # 8h for 1e5 turns * 1e6 particles = 3e-7 s/particle/turn  (saving 10 frames a 5 turns)

    sim.launch_tracking(nturns, save=output)
    sim.summary()
    sim.analyze_data(output, sg_samples, qSample,
                     sampling_freq, sim.frev, turns_init*samples_per_turn,
                     turnsPerExc*excSamples*samples_per_turn)
    sim.plot(save=output)

currentTime = datetime.datetime.now()

description = f"""
Simulation end
Date: {currentTime}
Time: {currentTime-initTime}
CP  : {(currentTime-initTime).total_seconds()} s
"""
print(description)
