"""
Exciter element

Author: Philipp Niedermayer
Date: 11.11.2022

"""


from pathlib import Path

import xobjects as xo
import xtrack as xt
import numpy as np


class Exciter(xt.BeamElement):
    """Beam element modeling a stripline exciter as a thin time-dependent multipole.
    
    The multipole components are given by an array of samples and a sampling frequency
    and may contain arbitrary scaling factors:
        
        k_nl(t) = k_nl * samples[i]
    
    For example, to compute samples for a sinusoidal excitation with frequency f_ex one
    would do: samples[i] = np.sin(2*np.pi*f_ex*i/sampling). 
    
    It is *not* assumed that the variations are slow compared to the revolution frequency
    and the particle arrival time is taken into account when determining the sample index:
    
        i = sampling * ( ( at_turn - start_turn ) / f_rev - zeta / beta0 / c0 )
    
    where zeta=(s-beta0*c0*t) is the longitudinal coordinate of the particle, beta0 the
    relativistic beta factor of the particle, c0 is the speed of light, at_turn is the
    current turn number, and f_rev is the revolution frequency. The excitation starts 
    with the first sample when the reference particle arrives at the element in start_turn.
    
    Notes:
        - This is not to be confused with an RFMultipole, which inherits the characteristics
          of an RFCavity and whose oscillation is therefore with respect to the reference
          particle. The frequency of the RFMultipole can therefore only be a harmonic of the
          revolution frequency.
        - This is also not to be confused with an ACDipole, for which the oscillation is 
          assumed to be slow compared to the revolution frequency and the kick is the same 
          for all particles independent of their longitudinal coordinate.    
        
        
    Parameters:
    
        - knl [m^-n, array]: Normalized integrated strength of the normal components.
        - ksl [m^-n, array]: Normalized integrated strength of the skew components.
        - samples [1, array]: Samples of excitation strength.
        - sampling [Hz]: Sampling frequency.
        - frev [Hz]: Revolution frequency of beam (required to relate turn number to sample index).
        - start_turn [1]: Start turn of excitation.
        
    """
    
    _xofields={
        'order': xo.Int64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'samples': xo.Float64[:],
        'nsamples': xo.Int64,
        'sampling': xo.Float64,
        'frev': xo.Float64,
        'start_turn': xo.Int64,
        }
    
    _extra_c_sources = [Path(__file__).parent.absolute().joinpath('exciter.h')]
    
    
    def __init__(self, *, samples=None, nsamples=None, sampling=0, frev=0, knl=[1], ksl=[], start_turn=0, **kwargs):
        
        # sanitize knl and ksl array length
        n = max(len(knl), len(ksl))
        nknl = np.zeros(n, dtype=np.float64)
        nksl = np.zeros(n, dtype=np.float64)
        if knl is not None:
            nknl[:len(knl)] = np.array(knl)
        if ksl is not None:
            nksl[:len(ksl)] = np.array(ksl)
        order = n - 1
        
        if samples is not None:
            if nsamples is not None and nsamples != len(samples):
                raise ValueError("Only one of samples or nsamples may be specified")
            nsamples = len(samples)
        if samples is None:
            samples = np.zeros(nsamples)
        
        super().__init__(order=order, knl=nknl, ksl=nksl, samples=samples, 
                         nsamples=nsamples, sampling=sampling, frev=frev,
                         start_turn=start_turn, **kwargs)

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        ctx2np = self._buffer.context.nparray_from_context_array
        return self.__class__(knl=-ctx2np(self.knl),
                              ksl=-ctx2np(self.ksl),
                              samples=self.samples,
                              nsamples=self.nsamples,
                              sampling=self.sampling,
                              frev=self.frev,
                              start_turn=self.start_turn,
                              _context=_context, _buffer=_buffer, _offset=_offset)
