#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demonstration of in-phase and quadrature (IQ) demodulation 
of a Vector Network Analyzer in the context of Beam Transfer Function (BTF) measurements

"""

__author__ = "Philipp Niedermayer"
__contact__ = "p.niedermayer@gsi.de"
__date__ = "2022-04-11"




import numpy as np



def magnitude_phase_difference(t, s1, s2, f):
    """
    :param t: array of sampling times
    :param s1: array of signal 1 real values
    :param s2: array of signal 2 real values
    :param f: frequency of interest
    :return: relative magnitude and phase of signal 2 compared to signal 1 at given frequency
             returned phase is in radians and in range [-pi, +pi]
    """
    # IQ demodulation
    sin = np.sin(2*np.pi*f*t)
    cos = np.cos(2*np.pi*f*t)
    C1 = np.trapz(s1*cos) + 1j*np.trapz(s1*sin)
    C2 = np.trapz(s2*cos) + 1j*np.trapz(s2*sin)    
    return np.abs(C2)/np.abs(C1), (np.angle(C1)-np.angle(C2)+np.pi)%(2*np.pi)-np.pi








if __name__ == '__main__':
    
    # BTF measurement example
    ##########################
    
    # the sampling times (in units of turn number)
    t = np.arange(2**16)
    
    A1 = 1
    A2 = 0.1
    phi1 = -0.5
    phi2 = 0.1234
    # the excitation signal (turn-by-turn kick)
    q_ex = 0.31
    s1 = A1*np.sin(2*np.pi*q_ex*t + phi1)
    
    mag   = A2/A1
    phase = phi2-phi1
    # the measured signal (turn-by-turn beam position from pickup)
    s2 = A2*np.sin(2*np.pi*q_ex*t + phi2) # ideal linear response
    s2 += 0.2*np.random.random(len(t)) # add some noise, other frequency components etc.
    s2 += 0.8*np.sin(2*np.pi*0.3*t + 0.56)    
    s2 += 2.5*np.cos(2*np.pi*0.311*t)
    
    print(' Real magnitude and phase ')
    print(f'Magnitude: {mag:g}')
    print(f'Phase:     {phase:g} rad')

    print(' Computed magnitude and phase with IQ demodulation')
    print(' Signal was added noise and other frequency components')
    # Network analyser S21
    mag, phase = magnitude_phase_difference(t, s1, s2, q_ex)
    print(f'Magnitude: {mag:g}')
    print(f'Phase:     {phase:g} rad')
    # mag and phase will correspond to the linear response of the system
