# -*- coding: utf-8 -*-

"""

BTF simulation based on Philipp's
slow extraction simulation codes


Requires xtrack>=0.40.0

"""

__author__ = "Philipp Niedermayer & CCo"
__contact__ = "edgar.cristopher.cortes.garcia@desy.de"
__date__ = "2022-12-09"


import xtrack as xt
import xobjects as xo
import xpart as xp
import numpy as np
import pandas as pd
import json
import os
import time
import datetime
from pint import Quantity as Qty
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib import pyplot as plt

try:
    from tqdm import tqdm
except:
    print('Failed to import tqdm. No progress will be reported')
    class tqdm():
        __init__ = __exit__ = update = lambda *a, **k: None
        __enter__ = lambda s, *a, **k: s
        write = lambda s, *a, **k: print(*a, **k)
        

from pint import Quantity as Qty
c_light = Qty('c')

from .ion import Ion
from .exciter import Exciter
from .centroid_monitor import CentroidMonitor
from .BPM import BPM
from .beam_position_monitor import BeamPositionMonitor

def npf(arr):
    """Copy array to CPU as necessary to make sure it's a numpy array"""
    if hasattr(arr, 'get'):
        return arr.get()
    return arr


    


def auto_context():
    import pyopencl as cl
    
    gpus = []
    print(f'Available platforms ({len(cl.get_platforms())}):')
    for ip, platform in enumerate(cl.get_platforms()):
        print(f'  {ip} {platform.name} ({platform.vendor})\n'
              f'    {platform.version}')
        for id, device in enumerate(platform.get_devices()):
            
            typ = "GPU" if device.type == cl.device_type.GPU else "CPU"  if device.type == cl.device_type.CPU else "???"
            print(f'    {ip}.{id} {typ}: {device.name}')
            if device.type == cl.device_type.GPU: # DEBUG: CPU+GPU
                gpus.append(f'{ip}.{id}')
        print()
    
    return {i: xo.ContextPyopencl(i) for i in gpus}
    #return {0: xo.ContextPyopencl(gpus[0])}

class Trackpart:
    """
    Class to maintain tracking of a particle subset on a single GPU context
    """
    
    def __init__(self, linetrack, particles, nturns):
        self.linetrack = linetrack
        self.particles = particles
        self.nparticles = len(self.particles.state)
        self.at_turn = 0
        self.nturns = nturns
    
    def track(self, until_turn=None, debug=False, progress=None):
        """Track until given turn or self.nturns"""
        
        if until_turn is None or until_turn > self.nturns:
            until_turn = self.nturns
        
        if self.at_turn == 0:  # just starting
            self.time = time.perf_counter()
        
        while self.at_turn < until_turn:
            n = until_turn-self.at_turn
            
            if progress:
                # provide progress feedback at powers of 10 or every ~5%
                n = max(1, min(n, 10**int(1+np.log10(max(self.at_turn,1)))-self.at_turn,
                               self.nturns/20))
            
            # track
            if debug:
                print(f'Tracking for {n} turns SKIPPED in DEBUG mode !!!!!!!!!!!!!!')
            else:
                self.linetrack.track(self.particles, num_turns=n)
                self.linetrack._context.synchronize() # mandatory for CuPy context to wait
                
            self.at_turn += n
            
            if progress: 
                progress.update(n*self.nparticles)
        
        if self.at_turn == self.nturns:  # all done
            self.time = datetime.timedelta(seconds=time.perf_counter() - self.time)
        

class BTFSimulation:
    
    def __init__(self, slim=False, description=None, useCPU=False):
        """
        Use slim=True for testing or if not tracking is intended
        """
        print()
        print('BTF Simulation')
        if slim: print('   D E B U G   M O D E   ')
        print('==========================\n')
        print(description)
        print()
        
        self.perf_counter = dict(setup=time.perf_counter())
        self.slim = slim
        self.monitor_names = []
        
        # create contexts

        self.useCPU = useCPU
        if not useCPU:
            self.contexts = {None: None} if self.slim else auto_context()
        else:
            self.contexts = {1:xo.ContextCpu()}

        self.tracked_turns = 0
        # flag to signalize if an init distribution has been computed
        self.initDist = None

        self.sextStrengths = None
        self.correct_chroma = False

    def load_lattice(self, json_file, septum_aperture_name):
        # load line        
        print('\nLattice')
        with open(json_file, 'r') as fid:
            loaded_dct = json.load(fid)
            
        self._line = xt.Line.from_dict(loaded_dct)
        self.septum_aperture_name = septum_aperture_name
        
        print('  source file:', json_file)
        print('  line length:', self._line.get_length())
        print('  septum name:', self.septum_aperture_name)
    
    
    def set_particle(self, ion, nparticles):
        print('\n')
        print(ion)
        
        self.ion = ion
        self.particle_ref = xp.Particles(mass0=ion.m.to('eV/c²').magnitude,
                                         q0=ion.q.to('e').magnitude,
                                         p0c=ion.p.to('eV/c').magnitude)
        self._line.particle_ref = self.particle_ref
        self.nparticles = int(nparticles)
    
    
    @property
    def frev(self):
        return npf(self.particle_ref.beta0)[0]*299792458/self._line.get_length()
    
    def install_exciter(self, exciter_name, k0l, sampling, *, k0l_max=np.inf,
                              signal_file=None, channel=None, start_turn=0,
                              samples=None):
        """
        Install k0l stripline exciter
        
        Args:
            signal_file (str): Path to file containing output from GNU Radio (binary signal stream as complex64)
            sampling (float): Sampling frequency in Hz
            channel (int, optional): RF channel (1 for real and 2 for imaginary part of complex signal)
            
        """
        
        print('\nExcitation')
        
        self.exciter_name = exciter_name + '_exciter'
        self.exciter_k0l = k0l
        self.exciter_k0l_max = k0l_max
        if samples is None:
            if signal_file.endswith('.float32'):
                samples = np.copy(np.memmap(signal_file, np.float32))
            elif channel is not None:                                
                samples = np.copy(getattr(np.memmap(signal_file, np.complex64), {1:'real', 2:'imag'}[channel]))
            else:
                raise ValueError('Supply a .float32 file or specify channel number!')

        elif signal_file is not None:
                raise ValueError('Parameters signal_file and samples are mutually exclusive')
            
        frev = self.frev
        
        print('  name:', self.exciter_name)
        print('  k0l: ', min(self.exciter_k0l, self.exciter_k0l_max), 'rad * waveform')
        print('        mean:', min(self.exciter_k0l, self.exciter_k0l_max) * np.mean(samples), 'rad')
        print('        std:', min(self.exciter_k0l, self.exciter_k0l_max) * np.std(samples), 'rad')
        print('        min:', min(self.exciter_k0l, self.exciter_k0l_max) * np.min(samples), 'rad')
        print('        max:', min(self.exciter_k0l, self.exciter_k0l_max) * np.max(samples), 'rad')
        print('  file:', signal_file)
        print('  fs:  ', sampling/1e3, 'kHz')
        print('  frev:', frev/1e3, 'kHz')

        self.exciter = Exciter(samples=samples, sampling=sampling, frev=frev, start_turn=start_turn,
                               knl=[min(self.exciter_k0l, self.exciter_k0l_max)])
        
        self._line.insert_element(
            index=exciter_name,
            name=self.exciter_name,
            element=self.exciter,
        )
    
    
    def install_monitor(self, *, at_element, start_at_turn=None, stop_at_turn=None,
                              n_repetitions=1, repetition_period=0, name=None, num_particles=None):
        name = name or f'monitor{len(self.monitor_names)}'
        if name == 'particles' or name in self.monitor_names:
            raise ValueError(f'Invalid name: {name}')
        num_particles = self.nparticles if num_particles is None else int(min(num_particles, self.nparticles))
        print('\nMonitor')
        print(f'  name:      {name}')
        print(f'  at:        {at_element}')
        print(f'  particles: {num_particles}')
        print(f'  range:     {start_at_turn} to {stop_at_turn} (excl.)')
        print(f'  every:     {repetition_period} ({n_repetitions} times)')
        print( '  turns:    ', f'{stop_at_turn-start_at_turn}*{n_repetitions}:', end=' ')
        for i in range(n_repetitions):
            if i < 3 or i > n_repetitions-3:
                o = start_at_turn + i*repetition_period
                print(f'{start_at_turn + i*repetition_period}' + 
                     (f'..{stop_at_turn-1 + i*repetition_period}' if stop_at_turn-start_at_turn>1 else ''), end=' ')
            if i == 3:
                print('...', end=' ')
        print()
        
        print('  Memory:   ', f'~{40e-6*num_particles*(stop_at_turn-start_at_turn)*n_repetitions:.0f} MB')
        
        if self.slim:
            print('  DEBUG MODE: Buffer size set to 0!')
        
        self.save_to_mon_nparticles = num_particles
        # insert placeholder into line such that the structure (indices) remain consistent
        # while it will later be replaced by dedicated monitors for each context to track on
        placeholder_monitor = xt.ParticlesMonitor(
            start_at_turn=start_at_turn,
            stop_at_turn=stop_at_turn,
            n_repetitions=n_repetitions,
            repetition_period=repetition_period,
            num_particles=num_particles
        )
        placeholder_monitor._real_num_particles = num_particles
        self._line.insert_element(index=at_element, element=placeholder_monitor, name=name)
        self.monitor_names.append(name)
    
    def install_monitor_centroid(self, *, at_element, start_at_turn=0,
                                 stop_at_turn=None, name=None):
        """Install a "centroid_monitor" to save centroid data"""
        name = name or f'monitor{len(self.monitor_names)}'
        if name == 'particles' or name in self.monitor_names:
            raise ValueError(f'Invalid name: {name}')
        if stop_at_turn is None:
            raise ValueError(f'Stop at turn not provided')
        print('\nCentroid monitor')
        print(f'  name:        {name}')
        print(f'  at:          {at_element}')
        print(f'  start turn:  {start_at_turn}')
        print(f'  stop turn:   {stop_at_turn}')
        print( '  memory:   ', f'<{40e-3*(stop_at_turn-start_at_turn):.0f} KB')
        
        if self.slim:
            print('  DEBUG MODE: Buffer size set to 0!')
        
        # insert placeholder into line such that the structure (indices) remain consistent
        # while it will later be replaced by dedicated monitors for each context to track on
        
        placeholder_monitor = CentroidMonitor(
            start_at_turn=start_at_turn,
            stop_at_turn=stop_at_turn
        )
        self._line.insert_element(index=at_element, element=placeholder_monitor, name=name)
        self.monitor_names.append(name)

    
    def install_BeamPositionMonitor(self, *,
                                   particle_id_start=0,
                                   num_particles=None,
                                   start_at_turn=0,
                                   stop_at_turn=None,
                                   rev_frequency=1,
                                   sampling_frequency,
                                   at_element, name=None):
        """Install a BeamPositionMonitor to save centroid data"""
        name = name or f'monitor{len(self.monitor_names)}'
        if name == 'particles' or name in self.monitor_names:
            raise ValueError(f'Invalid name: {name}')
        if stop_at_turn is None:
            raise ValueError(f'Stop at turn not provided')
        print('\nBeamPositionMonitor')
        print(f'  name:        {name}')
        print(f'  at:          {at_element}')
        print(f'  start turn:  {start_at_turn}')
        print(f'  stop turn:   {stop_at_turn}')
        print(f'  samples per turn : {int(sampling_frequency/rev_frequency)}')
        print( '  memory:   ', f'~ 3 * {40e-6*(stop_at_turn-start_at_turn):.0f} MB')
        print() 
        if self.slim:
            print('  DEBUG MODE: Buffer size set to 0!')
        
        # insert placeholder into line such that the structure (indices) remain consistent
        # while it will later be replaced by dedicated monitors for each context to track on
        placeholder_monitor = BeamPositionMonitor(particle_id_start=particle_id_start,
                                                  num_particles=num_particles,
                                                  start_at_turn=start_at_turn,
                                                  stop_at_turn=stop_at_turn,
                                                  frev=rev_frequency,
                                                  sampling_frequency=sampling_frequency)
        
        self._line.insert_element(index=at_element,
                                  element=placeholder_monitor,
                                  name=name)
        self.monitor_names.append(name)

    def _make_tracker(self, context, line):
        """Returns a line with a tracker optimized for tracking"""
        line1 = line.copy(_context=context)
        
        line1.reset_s_at_end_turn = True
        
        line1.build_tracker(_context=context)
        line1.optimize_for_tracking(compile=True, verbose=False)

        return line1

    def complete_line(self):
        print('\nBuilding default tracker')
        
        line = self._line

        #s0 = line.get_s_position(self.septum_aperture_name)
        s0 = 0
        
        print(f'   Inserting septum at {s0:g} m\n')
        self.septum_entrance_aperture = 80e-3 
        septum_entrance_aperture = self.septum_entrance_aperture
        # Insert septum with aperture (don't want to do this via MAD-X import
        # due to https://github.com/xsuite/xtrack/issues/247)

        #min_x = septum_entrance_aperture if septum_entrance_aperture < 0 else -np.inf
        #max_x = septum_entrance_aperture if septum_entrance_aperture > 0 else np.inf
        max_x = septum_entrance_aperture
        min_x = -septum_entrance_aperture
        
        min_y = -25e-3 
        max_y =  25e-3

        line.insert_element(
            at_s=s0,
            name='septum_begin_collimator',
            element=xt.apertures.LimitRect(min_x=min_x, max_x=max_x,
                                           min_y=min_y, max_y=max_y),
        )

        # CPU tracker for simple tasks like twiss etc.
        self.linetrack = self._make_tracker(None, line)
        
        print('\nDefault tracker')
        print(f'  nelements: {len(self.linetrack.elements)}')
    
    
    @property
    def line(self):
        """The line used for tracking (optimized for tracking)"""
        return self.linetrack
        
    
    def create_beam(self, *, emitt_x, emitt_y,
                    rel_momentum_spread, save=None, bunched=False):
        """Create particle beam (at s=0)"""
        
        print('\nBeam')
        print('  nparticles:    ', self.nparticles)
        print('  emitt_x:       ', emitt_x)
        print('  emitt_y:       ', emitt_y)
        print('  bunched:       ', bunched)

        emx = emitt_x.magnitude*1e-6
        emy = emitt_y.magnitude*1e-6

        ## Transverse
        
        twiss = self.linetrack.twiss(method='4d')

        # Twiss parameters
        betx0 = twiss.betx[0]
        alfx0 = twiss.alfx[0]
        gamx0 = twiss.gamx[0]
        Dx0 = twiss.dx[0]
        Dpx0 = twiss.dpx[0]

        bety0 = twiss.bety[0]
        alfy0 = twiss.alfy[0]
        gamy0 = twiss.gamy[0]
        Dy0 = twiss.dy[0]
        Dpy0 = twiss.dpy[0]

        x0 = twiss.x[0]
        px0 = twiss.px[0]

        Trev0 = twiss.T_rev0

        print()
        print("Horizontal tune is",twiss.qx)
        print('    Optical functions at beginning')
        print()
        print(f'   betx0 : {betx0:g} m, alfx0 : {alfx0:g}')
        print(f'   bety0 : {bety0:g} m, alfy0 : {alfy0:g}')
        print(f'   Dx0   : {Dx0:g}   m, Dpx0  : {Dpx0:g}')
        print(f'   Dy0   : {Dy0:g}   m, Dpy0  : {Dpy0:g}')
        print(f'   x0    : {x0:g}    m, px0   : {px0:g}')
        print()
        print('Revolution frequency calculated', 1/Trev0)
        print()

        print(' Tunes')
        print(f'Qx : {twiss.qx:g}')
        print(f'Qy : {twiss.qy:g}')
        print(' Chroma')
        print(f'dQx : {twiss.dqx:g}')
        print(f'dQy : {twiss.dqy:g}')

        sigx = emx*np.array([[betx0, -alfx0],
                             [-alfx0, gamx0]])

        sigy = emy*np.array([[bety0, -alfy0],
                             [-alfy0, gamy0]])

        x0,px0=twiss.x[0],twiss.px[0]
        y0,py0=twiss.y[0],twiss.py[0]        

        x, px = np.random.multivariate_normal([x0, px0], sigx, self.nparticles).T
        y, py = np.random.multivariate_normal([y0, py0], sigy, self.nparticles).T

        
        ## Longitudinal
        if bunched:
            # Bunched beam (matched to bucket, see also below)
            raise NotImplementedError('Not tested yet')
            zeta, delta = xp.generate_longitudinal_coordinates(
                particle_ref=particle_ref,
                num_particles=self.nparticles,
                distribution='gaussian',
                engine="single-rf-harmonic", # or pyheadtail if installed
                #rf_voltage=0, rf_harmonic=1, rf_phase=0, momentum_compaction_factor=,
                # default uses these from our tracker/line
                sigma_z=10e-2, # TODO: rms_bunch_length Delta-zeta/zeta
            )

        else:
            # Coasting beam (uniform in zeta, gaussian in delta)
            print('  rel_momentum_spread:', rel_momentum_spread)
            
            zeta = self.line.get_length()*np.random.uniform(-0.5, 0.5, self.nparticles) ## Set to zero
            delta = rel_momentum_spread.to('').magnitude*xp.generate_2D_gaussian(self.nparticles)[0]  ## set to zero

        self.particles = xp.build_particles(
            #_context=self.context,
            particle_ref=self.particle_ref,
            x=x, px=px, 
            y=y, py=py,
            zeta=zeta, delta=delta
        )

    def _compute_sextupole_strengths(self, correct_chroma):

        """
        Compute the sextupole strengths for the HIT machine
        with the possibility of additional chromaticity correction
        """

        line = self.line
        kPrimeL = self.kPrimeL

        deg2rad = np.pi/180
        phic =  126*deg2rad     # Phase of the chromatic sextupoles
        phi0 = -106.06*deg2rad  # Arbitrary inital phase shift
        hs   =  5               # Arbitrary harmonic

        k2nl_S2KS1C = -kPrimeL*np.cos(hs*phic-phi0) # 1st sext for chromaticity control
        k2nl_S5KS3C =  kPrimeL*np.cos(hs*phic-phi0) # 2nd sext for chromaticity control
        k2nl_S3KS2R =  kPrimeL*np.cos(phi0) # 1st sext for driving 3rd order resonance
        k2nl_S6KS4R = -kPrimeL*np.cos(phi0) # 2nd sext for driving 3rd order resonance

        k2nlDict = {'S2KS1C':k2nl_S2KS1C,'S5KS3C':k2nl_S5KS3C,
                    'S3KS2R':k2nl_S3KS2R,'S6KS4R':k2nl_S6KS4R}
        
        for sextupole_key in k2nlDict:
            line[sextupole_key.lower()].knl[2] = k2nlDict[sextupole_key]

        if correct_chroma:
            
            self.correct_chroma = True
            print('\nCorrecting chromaticity\n')
            # Find sextupoles 
            sextupoles, k2l = [], []
            for name, el in zip(self.line.element_names, self.line.elements):
                if hasattr(el, "knl") and el.order >= 2:
                    sextupoles.append(name)
                    k2l.append(npf(el.knl[2]))
            # Twiss
            tw = line.twiss(method='4d', at_elements=sextupoles)
            betx, dx = tw.betx, tw.dx

            k2nl_S2KS1C -= tw.dqx*np.pi/betx[0]/dx[0]
            k2nl_S3KS2R -= tw.dqx*np.pi/betx[1]/dx[1]
            k2nl_S5KS3C -= tw.dqx*np.pi/betx[2]/dx[2]
            k2nl_S6KS4R -= tw.dqx*np.pi/betx[3]/dx[3]
            
        k2nlDict = {'S2KS1C':k2nl_S2KS1C,'S5KS3C':k2nl_S5KS3C,
                    'S3KS2R':k2nl_S3KS2R,'S6KS4R':k2nl_S6KS4R}

        if self.sextStrengths is None:
            self.sextStrengths = k2nlDict
        
        if correct_chroma:
            
            for sextupole_key in k2nlDict:
                line[sextupole_key.lower()].knl[2] = k2nlDict[sextupole_key]

            tw = line.twiss(method='4d', at_elements=sextupoles)
            print('\nTwiss after correction')
            print(f'Qx  = {tw.qx:g}')
            print(f'Qy  = {tw.qy:g}')
            print(f'dQx = {tw.dqx:g}')
            print(f'dQy = {tw.dqy:g}')
            print()

    def set_sextupole_str(self, kPrimeL, line=None, correct_chroma=False): 
        """Sets the sextupole strengths given a kPrimeL as in the HIT control system"""
        if line is None:
            self.kPrimeL = kPrimeL
            self._compute_sextupole_strengths(correct_chroma)
            line = self.line

        k2nlDict = self.sextStrengths
        
        for sextupole_key in k2nlDict:
            line[sextupole_key.lower()].knl[2] = k2nlDict[sextupole_key]

        print(f'\n   Changed kPrimeL to {kPrimeL}\n')

            
    def determine_sextupole(self, line=None):
        """Determine virtual sextupole strength"""
        if line is None:
            line = self.line
       
        # Find sextupoles 
        sextupoles, k2l = [], []
        for name, el in zip(line.element_names, line.elements):
            if hasattr(el, "knl") and el.order >= 2 and el.knl[2]:
                sextupoles.append(name)
                k2l.append(npf(el.knl[2]))

        # Twiss
        tw = self.linetrack.twiss(method='4d', at_elements=sextupoles)

        betx, mux = tw.betx, tw.mux

        print('\nTwiss')
        #print('  '+'\n  '.join(str(tw.get_summary()).split('\n')))

        # virtual sextupole
        Sn = -1/2*betx**(3/2)*k2l
        Stotal = np.sum(Sn * np.exp(3j*mux*2*np.pi))
        self.S = np.abs(Stotal)
        self.Smu = np.angle(Stotal)/3/2/np.pi
        
        print('\nSextupoles')
        print('  ' + '\n  '.join(str(pd.DataFrame(dict(name=sextupoles, k2l=k2l,
                                                       betx=betx, mux=mux))).split('\n')))
        print('  ------------------------------------------')
        print('  Virtual sextupole:', f'S = {self.S:g} at mu = {self.Smu:g}')

    def set_closed_orbit_correctors(self):
        # Werte vom 18. Dezember 2020 
        # E18 Kohlenstoff 6+
        # AX_S1MU1A_COC_EXTR = -1.5 # mrad
        # AX_S2MU1A_COC_EXTR =  0.5 # mrad
        # AX_S3MU1A_COC_EXTR =  0.3 # mrad
        # AX_S4MU1A_COC_EXTR = -0.7 # mrad
        # AX_S5MU1A_COC_EXTR = -0.017  # mrad
        # AX_S6MU1A_COC_EXTR = -2   # mrad
        # AX_S1MS1V_COC_EXTR =  0.3 # mrad
        
        corStr = {'S1MU1A':-1.5*1e-3, 'S2MU1A':0.5*1e-3,
                  'S3MU1A': 0.3*1e-3, 'S4MU1A':-0.7*1e-3,
                  'S5MU1A':-0.017*1e-3,'S6MU1A':-2.0*1e-3,
                  'S1MS1V':0.3*1e-3
        }
        for corri in corStr:
            self._line[corri.lower()].knl[0] = -corStr[corri]

        print("\n   Closed orbit correctors set up\n")
        
    def finish_setup(self):
        print('\nSetup completed')


    def _replace_monitors(self, context, line, offset):
        """
        Here we only reinstall the centroid monitors for the BTF simulation
        """
        for name in self.monitor_names:
            placeholder_monitor = self._line.element_dict[name]
            #m = max(min(placeholder_monitor._real_num_particles-offset, n), 0)
            m = 0
            if isinstance(placeholder_monitor, xt.ParticlesMonitor):
                print(' Replacing particle monitor\n')
                print(f' start : {placeholder_monitor.start_at_turn}')
                print(f' stop  : {placeholder_monitor.stop_at_turn}\n')
                monitor = xt.ParticlesMonitor(_context=context, 
                                              start_at_turn=placeholder_monitor.start_at_turn,
                                              stop_at_turn=placeholder_monitor.stop_at_turn, 
                                              n_repetitions=placeholder_monitor.n_repetitions,
                                              repetition_period=placeholder_monitor.repetition_period,
                                              num_particles=self.save_to_mon_nparticles)
                
            elif isinstance(placeholder_monitor, CentroidMonitor):
                print(' Replacing centroid monitor\n')
                print(f' start : {placeholder_monitor.start_at_turn}')
                print(f' stop  : {placeholder_monitor.stop_at_turn}\n')
                monitor = CentroidMonitor(start_at_turn=placeholder_monitor.start_at_turn,
                                          stop_at_turn=placeholder_monitor.stop_at_turn,
                                          _context=context)
            elif isinstance(placeholder_monitor, BPM):
                print(' Replacing BPM\n')
                print(f' start : {placeholder_monitor.start_at_turn}')
                print(f' stop  : {placeholder_monitor.stop_at_turn}\n')
                print(f' rev freq : {placeholder_monitor.rev_frequency}\n')
                print(f' samp freq  : {placeholder_monitor.sampling_frequency}\n')
                monitor = BPM(particle_id_start=placeholder_monitor.particle_id_start,
                              num_particles=placeholder_monitor.num_particles,
                              start_at_turn=placeholder_monitor.start_at_turn,
                              stop_at_turn=placeholder_monitor.stop_at_turn,
                              rev_frequency=placeholder_monitor.rev_frequency,
                              sampling_frequency=placeholder_monitor.sampling_frequency,
                              _context=context)
            
            elif isinstance(placeholder_monitor, BeamPositionMonitor):
                print(' Replacing BeamPositionMonitor\n')
                print(f' start : {placeholder_monitor.start_at_turn}')
                print(f' stop  : {placeholder_monitor.stop_at_turn}')
                print(f' rev freq : {placeholder_monitor.frev*1e-6:g} MHz')
                print(f' samp freq  : {placeholder_monitor.sampling_frequency*1e-6:g} MHz\n')
                monitor = BeamPositionMonitor(particle_id_start=placeholder_monitor.particle_id_start,
                              num_particles=placeholder_monitor.num_particles,
                              start_at_turn=placeholder_monitor.start_at_turn,
                              stop_at_turn=placeholder_monitor.stop_at_turn,
                              frev=placeholder_monitor.frev,
                              sampling_frequency=placeholder_monitor.sampling_frequency,
                              _context=context)
            
            else:
                raise RuntimeError(f"Monitor {name} is not supported for BTF simulation: {placeholder_monitor}")
            line.element_dict[name] = monitor
        
    def _prepare_contexts_for_track(self):
        """
        For each context (available GPUs) the line is copied, the monitors are freshly installed
        within the new context and the particle bunch is sliced, such that the tracking load
        is shared among the available resources
        """
        trackparts = []
        offset = 0
        for i, (id, context) in enumerate(self.contexts.items()):
            n = int((len(self.particles.state) - offset)/(len(self.contexts) - i))
            print(f'  - Tracker {i} with {n} particles on {context.__class__.__name__}({id}) {context.device if hasattr(context, "device") else ""}')

            # prepare line
            line = self._line.copy(_context=context)

            # Sextupole strength has to be re-setted in new XSuite version
            self.set_sextupole_str(self.kPrimeL, line=line, correct_chroma=self.correct_chroma)
            self.determine_sextupole(line=line)
            
            # replace monitors
            self._replace_monitors(context, line, offset)

            # create tracker
            linetrack = self._make_tracker(context, line)

            # assign particle subset
            particles = self.particles.filter(slice(offset, offset+n)).copy(_context=context)
            
            trackparts.append(Trackpart(linetrack, particles, self.nturns))

            offset += n        
        
        print()
        self.perf_counter['track'] = time.perf_counter()
        
        return trackparts

    def _track_on_contexts(self, trackparts):
        """
        Parallelization routine on the available resources
        """
        barFmt = "{l_bar}{bar}| {n:.0f}/{total:.0f} turns ({rate_fmt}) | {remaining} > {eta:%a %H:%M}"
        print(f'Tracking started at {time.asctime()}...', flush=True)
        with tqdm(total=self.nturns*self.nparticles,
                  unit_scale=1/self.nparticles,
                  bar_format=barFmt,
                  unit="", smoothing=0.01, ascii=True) as progress:
        
            ## Track
            with ThreadPoolExecutor(max_workers=len(trackparts)) as ex:
                futures = []
                for tr in trackparts:
                    futures.append(ex.submit(tr.track,
                                             until_turn=self.nturns,
                                             debug=self.slim,
                                             progress=progress))

                for future in as_completed(futures):
                    pass # this waits for completion of distributed trackers...
                    i = futures.index(future)
                    if future.cancelled() or future.exception() is not None:
                        progress.write(f'Tracker {i} canceled or error!')
                
            # all turns tracked
            for i, trackpart in enumerate(trackparts):
                if trackpart.at_turn != self.nturns:
                    raise ValueError(f'Tracker {i} is at turn {trackpart.at_turn} but expected at turn {at_turn} = {self.nturns}!')
                else:
                    progress.write(f'Tracker {i} completed {trackpart.at_turn} turns')

        #self.turns_tracked += self.nturns
        
        #print(f'Tracking finished at {time.asctime()}')
        self.perf_counter['save'] = time.perf_counter()
        #self.perf_counter['track'] = datetime.timedelta(seconds=self.perf_counter['save'] - self.perf_counter['track'])
        #print(f'\nTracking completed in', self.perf_counter['track'], '\n', flush=True)
        #print(' !!! Warning: Tracking performance counter gives incorrect values!!!! ')
        print()

        
    def launch_tracking(self, nturns, save='output'):
        """
        Launches the tracking for the BTF
        """
        self.nturns = nturns
        print('\nPreparing initial distribution')
        print('  nparticles:', self.nparticles)
        print('  nturns:    ', self.nturns)
        print('  ntracker:  ', len(self.contexts))

        if save:
            print('Saving initial distribution data', flush=True)
            os.makedirs(save, exist_ok=True)
        
            # save particle data
            to_save = ['s', 'x', 'y', 'px', 'py', 'zeta', 'delta',
                       'particle_id', 'at_element', 'at_turn', 'state']
            fname = os.path.join(save, 'particles_initDist.npz')
            print(f'  Saving data to {fname} ...', flush=True, end=' ')
            np.savez_compressed(fname, **{a: npf(getattr(self.particles, a)) for a in to_save})

            print(f'saved {self.particles.s.size} records of shape {self.particles.s.shape}: {os.path.getsize(fname)/1024**2:.1f} MB', flush=True)

        # Prepare trackers : change context, set monitors and slice particle bunch
        trackparts = self._prepare_contexts_for_track()

        barFmt = "{l_bar}{bar}| {n:.0f}/{total:.0f} turns ({rate_fmt}) | {remaining} > {eta:%a %H:%M}"
        print(f'Tracking finished at {time.asctime()}')
        self.perf_counter['save'] = time.perf_counter()
        self.perf_counter['track'] = datetime.timedelta(seconds=self.perf_counter['save'] - self.perf_counter['track'])
        print(f'\nTracking completed in', self.perf_counter['track'], '\n', flush=True)

        # launch tracking
        ##############
        self._track_on_contexts(trackparts)
        
        # collect results
        ##################
        
        # merge particle data
        self.particles = xp.Particles.merge([tr.particles for tr in trackparts],
                                            _context=self.particles._context)

        if save:
            print('Saving final distribution data', flush=True)
            os.makedirs(save, exist_ok=True)
        
            # save particle data
            to_save = ['s', 'x', 'y', 'px', 'py', 'zeta', 'delta',
                       'particle_id', 'at_element', 'at_turn', 'state']
            fname = os.path.join(save, 'particles_finalDist.npz')
            print(f'  Saving data to {fname} ...', flush=True, end=' ')
            np.savez_compressed(fname, **{a: npf(getattr(self.particles, a)) for a in to_save})

            print(f'saved {self.particles.s.size} records of shape {self.particles.s.shape}: {os.path.getsize(fname)/1024**2:.1f} MB', flush=True)
            
            self.perf_counter['save'] = datetime.timedelta(seconds=time.perf_counter() - self.perf_counter['save'])
            print(f'\nSaving completed in', self.perf_counter['save'], '\n', flush=True)

            # merge and save monitor data
            ##############################

            print()
            print(self.monitor_names)
            print()
            
            self.collect_data_from_monitors(save, trackparts)

        self.initDist = True
        self.trackparts = trackparts

        
    def collect_data_from_monitors(self, save, trackparts):
        
        for name in self.monitor_names:
            placeholder_monitor = self._line.element_dict[name]
                
            if isinstance(placeholder_monitor, xt.ParticlesMonitor):
                ## ParticlesMonitor
                ##############################################################################
                to_save_float = ['s', 'x', 'px', 'zeta', 'delta']  # , 'y', 'py'
                to_save_int = ['particle_id', 'at_element', 'at_turn', 'state']
            
                turns = placeholder_monitor.stop_at_turn - placeholder_monitor.start_at_turn
                frames = placeholder_monitor.n_repetitions
                parts = placeholder_monitor._real_num_particles
                shape = (frames, parts, turns) if frames > 1 else (parts, turns)

                fname = os.path.join(save, f'{name}.meta')
                print(f'  Saving data to {fname}', flush=True)
                # metadata
                np.savez(fname, shape=shape, name=name, at_element=self.line.element_names.index(name),
                         start_at_turn=placeholder_monitor.start_at_turn, 
                         stop_at_turn=placeholder_monitor.stop_at_turn,
                         n_repetitions=placeholder_monitor.n_repetitions, 
                         repetition_period=placeholder_monitor.repetition_period)

                # data
                for k in to_save_float+to_save_int:
                    dtype = 'int32' if k in to_save_int else 'float32'
                    fname = os.path.join(save, f'{name}-{k}.{dtype}')
                    print(f'    Saving data to {fname:50}', flush=True, end=' ')

                    #monitor = tr.linetrack.element_dict[name]

                    #data_i = npf(getattr(monitor, k)).astype(dtype)
                    
                    #print()
                    #print(f'Saving {k}')
                    #print(f'Data length in mon : {len(data_i)}\n')
                    
                    #np.savez(fname, k=data_i)

                    data = np.memmap(fname.replace('*', k), shape=shape, mode='w+', dtype=dtype)

                    offset = 0
                    for tr in trackparts:
                        monitor = tr.linetrack.element_dict[name]
                        m = monitor.part_id_end - monitor.part_id_start
                        data[..., offset:offset+m, :] = npf(getattr(monitor, k)).astype(dtype)
                        offset += m
                        print('.', end=' ', flush=True)

                    data.flush()
                    print(f'saved {data.size} records of shape {data.shape}: {os.path.getsize(fname)/1024**2:.1f} MB', flush=True)
            elif isinstance(placeholder_monitor, CentroidMonitor):
                
                print(' Collecting data from TbT monitor')
                print()
                ## CentroidMonitor
                ##############################################################################
                to_save_float = ['x_cen', 'y_cen']
                to_save_int = ['at_turn']
            
                turns = placeholder_monitor.stop_at_turn - placeholder_monitor.start_at_turn
                shape = (len(to_save_float)+len(to_save_int), len(placeholder_monitor.x_cen))
                
                fname = os.path.join(save, f'{name}.meta')
                print(f'  Saving data to {fname}', flush=True)
                # metadata
                np.savez(fname, name=name, at_element=self.line.element_names.index(name))
                    
                # data
                for k in to_save_float+to_save_int:
                    dtype = 'int32' if k in to_save_int else 'float32'
                    fname = os.path.join(save, f'{name}-{k}.{dtype}')
                    print(f'    Saving data to {fname:50}', flush=True, end=' ')
                    print()
                    average_this = []
                    for tr in trackparts:
                        monitor = tr.linetrack.element_dict[name]
                        data_i = npf(getattr(monitor, k)).astype(dtype)
                        average_this.append(data_i)

                    data = np.mean(average_this, axis=0)
                    np.save(fname, data)
                    #print(f'saved {data.size} records of shape {data.shape}: {os.path.getsize(fname)/1024**2:.1f} MB',
                    #flush=True)
            elif isinstance(placeholder_monitor, BPM):
                print(' Collecting data from BPM')
                print()
                to_save_float = ['x_cen', 'y_cen']
                to_save_int = ['at_turn',
                               'summed_particles',
                               'last_particle_id']
    
            
                turns = placeholder_monitor.stop_at_turn - placeholder_monitor.start_at_turn
                print(f'Turns : {turns}')
                shape = (len(to_save_float)+len(to_save_int), len(placeholder_monitor.x_cen))
                print(f'Shape : {shape}')
                fname = os.path.join(save, f'{name}.meta')
                print(f'  Saving data to {fname}', flush=True)
                # metadata
                np.savez(fname, name=name, at_element=self.line.element_names.index(name))
                    
                # data
                for k in to_save_float+to_save_int:
                    dtype = 'int32' if k in to_save_int else 'float32'
                    fname = os.path.join(save, f'{name}-{k}.{dtype}')
                    print(f'    Saving data to {fname:50}', flush=True, end=' ')
                    print()
                    average_this = []
                    for tr in trackparts:
                        monitor = tr.linetrack.element_dict[name]
                        data_i = npf(getattr(monitor, k)).astype(dtype)
                        average_this.append(data_i)

                    data = np.mean(average_this, axis=0)
                    np.save(fname, data)

            elif isinstance(placeholder_monitor, BeamPositionMonitor):
                print(' Collecting data from BeamPositionMonitor')
                print()
                to_save_float = ['x_cen']#, 'y_cen']
                to_save_int   = []
            
                turns = placeholder_monitor.stop_at_turn - placeholder_monitor.start_at_turn
                print(f'Turns : {turns}')
                shape = (len(to_save_float)+len(to_save_int), len(placeholder_monitor.x_cen))
                print(f'Shape : {shape}')
                fname = os.path.join(save, f'{name}.meta')
                print(f'  Saving data to {fname}', flush=True)
                # metadata
                np.savez(fname, name=name, at_element=self.line.element_names.index(name))

                # data
                for k in to_save_float+to_save_int:
                    dtype = 'int32' if k in to_save_int else 'float32'
                    fname = os.path.join(save, f'{name}-{k}.{dtype}')
                    print(f'    Saving data to {fname:50}', flush=True, end=' ')
                    print()
                    average_this = []
                    for tr in trackparts:
                        monitor = tr.linetrack.element_dict[name]
                        data_i = npf(getattr(monitor, k)).astype(dtype)
                        average_this.append(data_i)

                    data = np.mean(average_this, axis=0)
                    np.save(fname, data)
                    
            else:
                ## ? Monitor
                ##############################################################################
                raise RuntimeError(f"Monitor {name} is of unknown type: {placeholder_monitor}")
            
                
    
    def summary(self):
        print('\nSummary')        
        # state [int]: It is <= 0 if the particle is lost, > 0 otherwise
        lost = npf(self.particles.state) <= 0
        at_ele = npf(self.particles.at_element)
        print(f'  {len(self.particles.state)} particles:')
        for i in np.unique(at_ele[~lost]):
            n = np.sum(at_ele[~lost] == i)
            print(f'  - alive: {n} at element {i}:', self.line.element_names[i])
        for i in np.unique(at_ele[lost]):
            n = np.sum(at_ele[lost] == i)
            print(f'  - lost: {n} at element {i}:', self.line.element_names[i])
        n = np.sum(lost)
        print(f'  Total lost: {n} ({100*n/len(self.particles.state):.1f}%)')

    def plot(self, save="output"):
        print('\nGenerating plots...')
        
        os.makedirs(save, exist_ok=True)
        
        data = np.load(save+'/particles_initDist.npz')

        x = data['x']
        pid = data['particle_id']
        at_turn = data['at_turn']
        turns = np.unique(data['at_turn'])
        last_turn = turns[-1]

        plt.figure(dpi=200, figsize=(6,4))

        turnMask = last_turn == at_turn

        px  = data['px']
        dpp = data['delta']

        x  = x[turnMask]
        px = px[turnMask]
        dpp =dpp[turnMask]
        
        plt.scatter(x*1e3, px*1e3, c=dpp, s=3)
        plt.ylabel(r'p$_x [10^{-3}$]')
        plt.xlabel('x [mm]')
        plt.grid()
        plt.tight_layout()
        
        plt.savefig(save+'/initDist.png')
        
        print('', flush=True)
        data = np.load(save+'/particles_finalDist.npz')

        x = data['x']
        pid = data['particle_id']
        at_turn = data['at_turn']
        turns = np.unique(data['at_turn'])
        last_turn = turns[-1]

        plt.figure(dpi=200, figsize=(6,4))

        turnMask = last_turn == at_turn

        px  = data['px']
        dpp = data['delta']

        x  = x[turnMask]
        px = px[turnMask]
        dpp =dpp[turnMask]

        plt.scatter(x*1e3, px*1e3, c=dpp, s=3)
        plt.ylabel(r'p$_x [10^{-3}$]')
        plt.xlabel('x [mm]')
        plt.grid()
        plt.tight_layout()
        plt.savefig(save+'/finalDist.png')


    def analyze_data(self, save, exc_signal, tune_samples,
                     fs, frev, init_samples,
                     excited_samples, displayPlot=False):
        """
        Analyzes the centroid data given the external excitation signal
        and the tune samples
        """
        from .btf_analysis_tools import magnitude_phase_difference

        samples_per_turn = int(fs/frev)
        
        x_cen = np.load(save+'/BPM-Schottky-x_cen.float32.npy')
        
        turns = np.linspace(0, excited_samples/samples_per_turn, excited_samples)

        x_cen_analyze = x_cen[init_samples:init_samples+excited_samples] 
        
        from scipy import signal
        from matplotlib import pyplot as plt

        nSteps = len(tune_samples)
        x_cen_analyze = np.split(x_cen_analyze, nSteps)
        turns_analyze = np.split(turns, nSteps)

        exc_signal = np.split(exc_signal, nSteps)

        mag,phi = [],[]
        
        for i in range(nSteps):

            tune_i = tune_samples[i]

            s1_i = exc_signal[i]
            s2_i = x_cen_analyze[i]

            turns_i = turns_analyze[i]

            assert len(s1_i) == len(s2_i)
            # Network analyser S21
            mag_i, phase_i = magnitude_phase_difference(turns_i,
                                                        s1_i, s2_i,
                                                        tune_i)
            mag.append(mag_i)
            phi.append(phase_i)

        #cmap = cm.get_cmap('gist_heat')
        #phi = np.unwrap(np.array(phi))
        phi = np.array(phi)
        
        np.save(save+'/mag_phase.npy', [tune_samples, mag, phi])
        
        fig, ax = plt.subplots(2,1,dpi=200, figsize=(6,4),sharex=True,
                               gridspec_kw={'height_ratios': [2, 1], 'hspace':0})
        
        ax[0].plot(tune_samples, mag, marker='.')
        ax[1].plot(tune_samples, phi, marker='.')

        ax[1].set_xlabel(r'Excitation frequency $f_{\mathrm{exc}}/f_{\mathrm{rev}}$')
        ax[0].set_ylabel(r'Magnitude')
        ax[1].set_ylabel(r'Phase')

        plt.tight_layout()

        plt.savefig(save+'/btf_out_linscale.png')

        ax[0].set_yscale('log')
        
        plt.savefig(save+'/btf_out_logscale.png')
        
        ax[0].set_yscale('linear')

        if displayPlot:
            plt.show()
